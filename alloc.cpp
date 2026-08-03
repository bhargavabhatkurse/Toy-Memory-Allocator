// ============================================================================
//  alloc.cpp -- a small memory allocator, built up in three steps.
//
//  I wanted to understand how malloc/free actually work under the hood, so I
//  built three versions of an allocator. Each version fixes the main weakness
//  of the one before it:
//
//    Rung 1  Bump      : just move a pointer forward on each allocation.
//                        Very fast, but free() does nothing, so memory is
//                        never reused.
//    Rung 2  Implicit  : every block stores its size in a header and footer,
//                        so free() can reclaim memory and merge neighbouring
//                        free blocks. Downside: an allocation has to scan
//                        every block in the heap to find a free one.
//    Rung 3  Explicit  : keep the free blocks on a linked list, so an
//                        allocation only looks at free blocks instead of all
//                        of them. This is the fast version.
//
//  All three allocate out of one big region of memory that we grab from the OS
//  once (an "arena"). That way the benchmark measures the allocator's own
//  logic and not the operating system's page-fault handling.
//
//  Build: make
//  Run:   ./alloc --test     (checks correctness)
//         ./alloc --bench     (measures speed and memory usage)
// ============================================================================
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <sys/mman.h>

// Every payload we hand back is aligned to 16 bytes, because a lot of hardware
// prefers (or requires) aligned memory.
static const size_t ALIGNMENT = 16;

// Round n up to the next multiple of a. (a must be a power of two.)
static size_t align_up(size_t n, size_t a) {
    return (n + a - 1) & ~(a - 1);
}

// ----------------------------------------------------------------------------
//  Arena: one big block of memory we ask the OS for a single time. Each
//  allocator carves all of its allocations out of this region.
// ----------------------------------------------------------------------------
struct Arena {
    uint8_t* base = nullptr;
    size_t   size = 0;

    void map(size_t bytes) {
        size = bytes;
        base = (uint8_t*)mmap(nullptr, bytes,PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        if (base == MAP_FAILED) {
            perror("mmap");
            std::exit(1);
        }
    }

    void unmap() {
        if (base != nullptr) {
            munmap(base, size);
            base = nullptr;
        }
    }
};

// ----------------------------------------------------------------------------
//  Rung 1 -- Bump allocator.
//  The simplest thing that works: keep an offset into the arena and push it
//  forward on every allocation. It cannot reuse memory, so it is the speed
//  ceiling but the worst case for memory usage.
// ----------------------------------------------------------------------------
class BumpAllocator {
    Arena  arena;
    size_t offset = 0;   // how many bytes of the arena we have used so far

public:
    explicit BumpAllocator(size_t bytes) { arena.map(bytes); }
    ~BumpAllocator() { arena.unmap(); }

    void* alloc(size_t n) {
        if (n == 0) n = 1;
        n = align_up(n, ALIGNMENT);
        if (offset + n > arena.size) return nullptr;   // out of room
        void* p = arena.base + offset;
        offset += n;
        return p;
    }

    // Bump never reclaims memory, so free intentionally does nothing.
    void free(void*) {}

    void   reset()            { offset = 0; }
    size_t high_water() const { return offset; }
    double avg_scan()  const  { return 0.0; }   // it never searches
    static const char* name() { return "1_bump"; }
};

// ----------------------------------------------------------------------------
//  Shared block format for rungs 2 and 3
//
//  Each block looks like this in memory:
//      [ header (8 bytes) ][ payload ... ][ footer (8 bytes) ]
//
//  The header and footer both store the same word: the block's size, with the
//  lowest bit reused as an "is this block allocated?" flag. This trick works
//  because block sizes are always multiples of 16, so the low 4 bits of a size
//  are always zero and we can safely borrow the lowest one.
//
//  Throughout, "bp" means the payload pointer -- the address alloc() returns
//  to the caller.
// ----------------------------------------------------------------------------
namespace bt {
    const size_t WSIZE  = 8;    // size of a header or footer
    const size_t DSIZE  = 16;   // two words
    const size_t MINBLK = 32;   // smallest block we allow

    // Read / write one size_t at a pointer. We go through memcpy instead of a
    // plain pointer cast so we don't run into C++'s strict-aliasing rules; the
    // compiler still turns each of these into a single load or store.
    size_t get(const void* p) {
        size_t v;
        std::memcpy(&v, p, WSIZE);
        return v;
    }

    void put(void* p, size_t v) { //writes 8B to pointer address
        std::memcpy(p, &v, WSIZE);
    }

    // Combine a size and an allocated bit into one word, and read them back.
    size_t pack(size_t size, int allocated) {
        return size | (size_t)allocated;
    }

    size_t size_of(const void* p) {
        return get(p) & ~(size_t)0xF;   // clear the low 4 bits to get the size
    }
    int alloc_of(const void* p) {
        return (int)(get(p) & 1);       // the low bit is the allocated flag
    }

    // Pointer arithmetic to find parts of a block, or the neighbouring blocks.
    void* header(void* bp) {
        return (uint8_t*)bp - WSIZE;
    }
    void* footer(void* bp) {
        return (uint8_t*)bp + size_of(header(bp)) - DSIZE;
    }
    void* next_block(void* bp) {
        return (uint8_t*)bp + size_of((uint8_t*)bp - WSIZE);
    }
    void* prev_block(void* bp) {
        return (uint8_t*)bp - size_of((uint8_t*)bp - DSIZE);
    }

    // How large a block do we need to satisfy a request of `req` bytes?
    // (Add room for header + footer, round up, and never go below MINBLK.)
    size_t block_size_for(size_t req) {
        size_t b = align_up(req + DSIZE, DSIZE);
        if (b < MINBLK) b = MINBLK;
        return b;
    }

    // The heap starts with a small "prologue" and ends with an "epilogue",
    // both permanently marked as allocated. This means every real block always
    // has an allocated neighbour on each edge, so when we merge free blocks we
    // never have to special-case the very first or very last block.
    struct Heap {
        uint8_t* base = nullptr; //where arena begins
        void*    first_bp = nullptr; //pointer to the first block

        void init(uint8_t* b, size_t total_size) {
            base = b;

            put(base + 0, 0);                 // padding so payloads are 16-aligned
            put(base + 8,  pack(DSIZE, 1));   // prologue header 
            put(base + 16, pack(DSIZE, 1));   // prologue footer
            //(therefore the size is 16B of the first dummy block)

            // Everything left over becomes one big free block.
            size_t free_size = (total_size - 32) & ~(size_t)0xF;
            uint8_t* bp = base + 32;
            first_bp = bp;

            //one big block
            put(header(bp), pack(free_size, 0));
            put(footer(bp), pack(free_size, 0));


            put(base + 24 + free_size, pack(0, 1));   // epilogue header (size 0)
        }
    };
}

// ----------------------------------------------------------------------------
//  Rung 2 -- Implicit free list.
//  free() now reclaims memory and merges adjacent free blocks. But allocation
//  is first-fit over EVERY block, including allocated ones, so it is O(number
//  of blocks in the heap).
// ----------------------------------------------------------------------------
class ImplicitAllocator {
    Arena     arena;
    bt::Heap  heap;
    //for benchmarking
    size_t    high_water_  = 0;
    uint64_t  scanned_     = 0;   // total blocks visited across all allocs
    uint64_t  alloc_calls_ = 0;   // number of alloc() calls

public:
    explicit ImplicitAllocator(size_t bytes) { arena.map(bytes); reset(); }
    ~ImplicitAllocator() { arena.unmap(); }

    void reset() {
        heap.init(arena.base, arena.size); //converts to valid heap
        high_water_  = 0;
        scanned_     = 0;
        alloc_calls_ = 0;
    }

    size_t high_water() const { return high_water_; }

    double avg_scan()  const {
        return alloc_calls_ ? (double)scanned_ / alloc_calls_ : 0.0;
    }

    static const char* name() { return "2_implicit"; }

    void* alloc(size_t req) {
        using namespace bt;
        size_t needed = block_size_for(req); //beacuse we need to consider header+footer also
        alloc_calls_++;

        // First-fit: walk every block from the start until the epilogue (which
        // has size 0), and take the first free block that is big enough.
        for (void* bp = heap.first_bp; size_of(header(bp)) > 0; bp = next_block(bp)) {
            scanned_++;
            bool is_free = !alloc_of(header(bp));
            bool big_enough = size_of(header(bp)) >= needed;

            if (is_free && big_enough) {
                place(bp, needed);
                size_t end = (uint8_t*)bp + size_of(header(bp)) - arena.base;
                if (end > high_water_) high_water_ = end;
                return bp;
            }
        } 
        return nullptr;   // no room (fixed arena, we don't ask the OS for more)
    }

    void free(void* bp) {
        using namespace bt;
        if (bp == nullptr) return;
        size_t size = size_of(header(bp));
        put(header(bp), pack(size, 0));   // clear the allocated bit 1 to 0
        put(footer(bp), pack(size, 0));
        coalesce(bp);
    }

private:
    // Put an allocation into a free block, splitting off the leftover if the
    // remainder is large enough to be a block of its own.
    void place(void* bp, size_t needed) {
        using namespace bt;
        size_t block = size_of(header(bp));

        if (block - needed >= MINBLK) { //if we can split it into leftover block
            put(header(bp), pack(needed, 1));
            put(footer(bp), pack(needed, 1));

            void* rest = next_block(bp);
            put(header(rest), pack(block - needed, 0));
            put(footer(rest), pack(block - needed, 0));
        } 
        else {
            put(header(bp), pack(block, 1));
            put(footer(bp), pack(block, 1));
        }
    }

    // Merge a newly-freed block with any free neighbours. There are four cases
    // depending on whether the previous and next blocks are allocated.
    void coalesce(void* bp) {
        using namespace bt;
        bool prev_alloc = alloc_of(footer(prev_block(bp)));
        bool next_alloc = alloc_of(header(next_block(bp)));

        size_t size = size_of(header(bp));

        if (prev_alloc && next_alloc) {
            return;   // both neighbours in use: nothing to merge
        } else if (prev_alloc && !next_alloc) {
            // merge with the block after us
            size += size_of(header(next_block(bp)));

        } else if (!prev_alloc && next_alloc) {
            // merge with the block before us; the merged block starts there
            size += size_of(header(prev_block(bp)));
            bp = prev_block(bp);

        } else {
            // both neighbours free: merge all three into one
            size += size_of(header(prev_block(bp))) + size_of(header(next_block(bp)));
            bp = prev_block(bp);
        }

        put(header(bp), pack(size, 0));
        put(footer(bp), pack(size, 0));
    }
};

// ----------------------------------------------------------------------------
//  Rung 3 -- Explicit free list.
//  Free blocks are kept on a doubly linked list. The two pointers (prev, next)
//  are stored inside the free block's own payload -- that space is unused while
//  the block is free, so it costs no extra memory. Allocation now walks only
//  the free list instead of the whole heap.
// ----------------------------------------------------------------------------
class ExplicitAllocator {
    Arena     arena;
    bt::Heap  heap;
    void*     free_list    = nullptr;   // head of the free-block list
    size_t    high_water_  = 0;
    uint64_t  scanned_     = 0;
    uint64_t  alloc_calls_ = 0;

    // The prev/next pointers live at the start of a free block's payload.
    static void* get_prev(void* bp) {
        void* p;
        std::memcpy(&p, bp, sizeof p); //prev is in the first 8 bytes
        return p;
    }
    static void* get_next(void* bp) {
        void* p;
        std::memcpy(&p, (uint8_t*)bp + 8, sizeof p); // //next is in the next 8 bytes
        return p;
    }
    static void set_prev(void* bp, void* v) { //Writes a pointer into payload.
        std::memcpy(bp, &v, sizeof v);
    }
    static void set_next(void* bp, void* v) { //Writes a pointer into payload.
        std::memcpy((uint8_t*)bp + 8, &v, sizeof v);
    }

    // Add a block to the front of the free list.
    void insert(void* bp) {
        set_prev(bp, nullptr);
        set_next(bp, free_list);
        if (free_list != nullptr) set_prev(free_list, bp);
        free_list = bp; 
    }

    // Unlink a block from the free list.
    void remove(void* bp) {
        void* prev = get_prev(bp);
        void* next = get_next(bp);
        if (prev != nullptr) set_next(prev, next);
        else free_list = next; //removed block is the first block
        if (next != nullptr) set_prev(next, prev);
    }

public:
    explicit ExplicitAllocator(size_t bytes) { arena.map(bytes); reset(); }
    ~ExplicitAllocator() { arena.unmap(); }

    void reset() {
        heap.init(arena.base, arena.size);
        free_list = nullptr;
        insert(heap.first_bp);   // start with the one big free block on the list
        high_water_  = 0;
        scanned_     = 0;
        alloc_calls_ = 0;
    }
    size_t high_water() const { return high_water_; }
    double avg_scan()  const {
        return alloc_calls_ ? (double)scanned_ / alloc_calls_ : 0.0;
    }
    static const char* name() { return "3_explicit"; }

    void* alloc(size_t req) {
        using namespace bt;
        size_t needed = block_size_for(req);
        alloc_calls_++;

        // Walk only the free list, not every block in the heap.
        for (void* bp = free_list; bp != nullptr; bp = get_next(bp)) {
            scanned_++;
            if (size_of(header(bp)) >= needed) {
                place(bp, needed); //performs the allocation
                size_t end = (uint8_t*)bp + size_of(header(bp)) - arena.base;
                if (end > high_water_) high_water_ = end;
                return bp;
            }
        }
        return nullptr; //can't allocate
    }

    void free(void* bp) {
        using namespace bt;
        if (bp == nullptr) return;
        size_t size = size_of(header(bp));
        put(header(bp), pack(size, 0));
        put(footer(bp), pack(size, 0));
        coalesce(bp);
    }

private:
    void place(void* bp, size_t needed) {
        using namespace bt;
        size_t block = size_of(header(bp));
        remove(bp);   // this block is leaving the free list

        if (block - needed >= MINBLK) {
            put(header(bp), pack(needed, 1));
            put(footer(bp), pack(needed, 1));
            void* rest = next_block(bp);
            //give this new block its own header and footer
            put(header(rest), pack(block - needed, 0));
            put(footer(rest), pack(block - needed, 0));
            insert(rest);   // the leftover goes back onto the free list
        } else {
            put(header(bp), pack(block, 1));
            put(footer(bp), pack(block, 1));
        }
    }

    // Same four cases as before, but now we also have to keep the free list
    // correct: any neighbour we merge into must first be unlinked, and the
    // final merged block gets inserted at the end.
    void coalesce(void* bp) {
        using namespace bt;
        bool prev_alloc = alloc_of(footer(prev_block(bp)));
        bool next_alloc = alloc_of(header(next_block(bp)));
        size_t size = size_of(header(bp));

        if (prev_alloc && next_alloc) {
            // both neighbours in use: no merge, just add this block to the list
        } else if (prev_alloc && !next_alloc) {
            void* next = next_block(bp);
            remove(next); //remove next from the freelist
            size += size_of(header(next));
        } else if (!prev_alloc && next_alloc) {
            void* prev = prev_block(bp);
            remove(prev);
            size += size_of(header(prev));
            bp = prev;
        } else {
            void* prev = prev_block(bp);
            void* next = next_block(bp);
            remove(prev);
            remove(next);
            size += size_of(header(prev)) + size_of(header(next));
            bp = prev;
        }
        
        put(header(bp), pack(size, 0));
        put(footer(bp), pack(size, 0));
        insert(bp);
    }
};


// ============================================================================
//  Benchmark: one fixed workload of many small allocations and frees, with the
//  live set kept bounded. A reclaiming allocator keeps its memory footprint
//  near the live set; bump keeps growing because it never frees anything.
//
//  We report throughput and utilization (peak live bytes / footprint). The
//  seed is fixed so the workload is identical and reproducible for every rung.
// ============================================================================
struct Result {
    std::string name;
    double   mops;        // millions of operations per second
    double   scan;        // average blocks scanned per allocation
    double   util;        // peak live bytes / footprint
    size_t   peak_live;   // most bytes alive at the same time
    size_t   footprint;   // how far into the arena we reached
    uint64_t allocs;
    uint64_t frees;
};

template <class Alloc>
Result bench(size_t arena_size, int ops, uint64_t seed) {
    Alloc allocator(arena_size);
    std::mt19937_64 rng(seed); //so that every allocator gets the same workload

    struct Live { void* ptr; size_t size; }; //tracks every pointer that's currently allocated (not yet freed)
    std::vector<Live> live;
    live.reserve(4096); //to avoid vector scaling dynamically

    size_t   live_bytes = 0;
    size_t   peak = 0;
    uint64_t nalloc = 0;
    uint64_t nfree = 0;

    auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < ops; i++) {
        // Keep the live set under 2000 blocks so a reclaiming allocator gets
        // to actually reuse freed memory.

        if (live.size() < 2000 && (live.empty() || (rng() & 1))) { //decide whether to free or allocate
            size_t size = 16 + (rng() % 496);
            void* p = allocator.alloc(size);
            if (p == nullptr) continue;
            nalloc++;  //How many allocations succeeded
            live.push_back({p, size});
            live_bytes += size;
            if (live_bytes > peak) peak = live_bytes; //peak: maximum amount of memory that was simultaneously allocated during the entire benchmark.
        }

        else {  //start freeing temperorarily till live < 2000
            size_t j = rng() % live.size();
            allocator.free(live[j].ptr);
            nfree++;
            live_bytes -= live[j].size;
            live[j] = live.back(); //copy the last 
            live.pop_back(); //pop the last
        }
    }

    auto end = std::chrono::steady_clock::now();

    double seconds   = std::chrono::duration<double>(end - start).count();
    size_t footprint = allocator.high_water();
    double util      = footprint ? (double)peak / (double)footprint : 0.0;

    Result r;
    r.name      = Alloc::name();
    r.mops      = (ops / 1e6) / seconds;
    r.scan      = allocator.avg_scan();
    r.util      = util;
    r.peak_live = peak;
    r.footprint = footprint;
    r.allocs    = nalloc;
    r.frees     = nfree;
    return r;
}

int main(int argc, char** argv) {
    std::string mode = (argc > 1) ? argv[1] : "--bench";

    const size_t   ARENA = 512ull << 20;   // 512 MiB per allocator
    const uint64_t SEED  = 0xC0FFEE;

    if (mode == "--bench") {
        const int OPS = 2'000'000;   // sized so bump still fits in the arena

        Result rs[] = {
            bench<BumpAllocator>(ARENA, OPS, SEED),
            bench<ImplicitAllocator>(ARENA, OPS, SEED),
            bench<ExplicitAllocator>(ARENA, OPS, SEED),
        };

        std::printf("Workload : %d ops | live set <= 2000 blocks | sizes 16..511 B | seed 0x%llX\n",
                    OPS, (unsigned long long)SEED);
        std::printf("Arena    : %zu MiB per allocator, mmap-backed\n\n", ARENA >> 20);

        std::printf("%-12s %9s %8s %11s %11s %11s %7s\n",
                    "rung", "Mops/s", "ns/op", "scan/alloc", "peak-live", "footprint", "util");
        std::printf("%-12s %9s %8s %11s %11s %11s %7s\n",
                    "----", "------", "-----", "----------", "---------", "---------", "----");
        for (auto& r : rs) {
            std::printf("%-12s %9.2f %8.1f %11.1f %8.1f KiB %8.2f MiB %6.1f%%\n",
                        r.name.c_str(), r.mops, 1000.0 / r.mops, r.scan,
                        r.peak_live / 1024.0, r.footprint / (1024.0 * 1024.0),
                        r.util * 100);
        }

        std::printf("\nallocs/frees: %llu / %llu  (identical trace for every rung)\n",
                    (unsigned long long)rs[0].allocs, (unsigned long long)rs[0].frees);
        std::printf(
            "legend: scan/alloc = blocks visited per allocation -- the search cost, and\n"
            "        the direct cause of the throughput gap (bump 0, implicit walks all\n"
            "        blocks, explicit walks only free ones).  footprint = arena high-water;\n"
            "        util = peak-live / footprint (bump never reclaims, so it balloons).\n");
        return 0;
    }

    std::printf("usage: %s [--test | --bench]\n", argv[0]);
    return 2;
}