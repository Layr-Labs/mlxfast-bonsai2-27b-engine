// PagedBlockFreeQueue.swift
//
// Intrusive free-page queue used by the paged KV allocator. Prefix-cache
// hits must be able to resurrect a zero-ref page from the middle in O(1),
// which a stack or Array.remove(at:) cannot provide.

import Foundation

/// Queue membership is stored in arrays indexed by physical page id. The
/// poison page is never inserted. Initial order is ascending page id so a
/// fresh pool still allocates physically adjacent pages.
struct PagedBlockFreeQueue {
    private static let none: Int32 = -1

    private var previous: [Int32]
    private var next: [Int32]
    private var present: [Bool]
    private(set) var first: Int32 = none
    private(set) var last: Int32 = none
    private(set) var count = 0

    init(pageCount: Int, excluding excluded: Int32, initiallyEmpty: Bool = false) {
        precondition(pageCount >= 0)
        previous = [Int32](repeating: Self.none, count: pageCount)
        next = [Int32](repeating: Self.none, count: pageCount)
        present = [Bool](repeating: false, count: pageCount)
        for page in 0 ..< Int32(pageCount) where !initiallyEmpty && page != excluded {
            append(page)
        }
    }

    /// Extend an uncommitted replacement queue; new addresses are absent
    /// until their native segment is installed in the same transaction.
    mutating func extend(to pageCount: Int) {
        precondition(pageCount >= present.count)
        let extra = pageCount - present.count
        previous.append(contentsOf: repeatElement(Self.none, count: extra))
        next.append(contentsOf: repeatElement(Self.none, count: extra))
        present.append(contentsOf: repeatElement(false, count: extra))
    }

    func contains(_ page: Int32) -> Bool {
        guard page >= 0, Int(page) < present.count else { return false }
        return present[Int(page)]
    }

    /// Snapshot in eviction/allocation order. Test and diagnostics only.
    var elements: [Int32] {
        var result: [Int32] = []
        result.reserveCapacity(count)
        var cursor = first
        while cursor != Self.none {
            result.append(cursor)
            cursor = next[Int(cursor)]
        }
        return result
    }

    mutating func popFirst() -> Int32? {
        guard first != Self.none else { return nil }
        let page = first
        remove(page)
        return page
    }

    mutating func remove(_ page: Int32) {
        let index = Int(page)
        precondition(index >= 0 && index < present.count && present[index])
        let before = previous[index]
        let after = next[index]
        if before == Self.none {
            first = after
        } else {
            next[Int(before)] = after
        }
        if after == Self.none {
            last = before
        } else {
            previous[Int(after)] = before
        }
        previous[index] = Self.none
        next[index] = Self.none
        present[index] = false
        count -= 1
    }

    /// Reclassify a queued page as immediately reusable. This is used when
    /// invalidating one page dissolves a multi-layer cache bundle and leaves
    /// its other zero-ref pages without any cache alias.
    mutating func moveToFront(_ page: Int32) {
        guard first != page else { return }
        remove(page)
        prepend(page)
    }

    mutating func append(_ page: Int32) {
        insert(page, before: Self.none)
    }

    mutating func prepend(_ page: Int32) {
        let index = Int(page)
        precondition(index >= 0 && index < present.count && !present[index])
        let oldFirst = first
        previous[index] = Self.none
        next[index] = oldFirst
        present[index] = true
        if oldFirst == Self.none {
            last = page
        } else {
            previous[Int(oldFirst)] = page
        }
        first = page
        count += 1
    }

    /// Preserve the supplied order at the head. Calling `prepend` repeatedly
    /// would reverse it, so install the batch back-to-front.
    mutating func prepend<S: Sequence>(contentsOf pages: S) where S.Element == Int32 {
        let materialized = Array(pages)
        for page in materialized.reversed() { prepend(page) }
    }

    mutating func append<S: Sequence>(contentsOf pages: S) where S.Element == Int32 {
        for page in pages { append(page) }
    }

    private mutating func insert(_ page: Int32, before successor: Int32) {
        let index = Int(page)
        precondition(index >= 0 && index < present.count && !present[index])
        precondition(successor == Self.none, "only tail insertion is supported")
        let oldLast = last
        previous[index] = oldLast
        next[index] = Self.none
        present[index] = true
        if oldLast == Self.none {
            first = page
        } else {
            next[Int(oldLast)] = page
        }
        last = page
        count += 1
    }
}
