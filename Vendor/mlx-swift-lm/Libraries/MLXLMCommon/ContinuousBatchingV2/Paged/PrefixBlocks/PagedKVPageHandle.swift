// PagedKVPageHandle.swift
//
// Generation-checked identities for physical KV pages. A bare page id is
// insufficient once zero-ref cached pages can be recycled: generation makes
// stale cache metadata fail closed instead of aliasing newly written KV.

import Foundation

struct PagedKVPageHandle: Hashable, Sendable {
    let group: PagedKVGroupKey
    let page: Int32
    let generation: UInt64
}

/// The prefix index is non-owning. The pool asks it whether a zero-ref page
/// is still cache-visible (free-queue placement) and tells it to remove all
/// aliases before that page is handed out for a write.
protocol PagedKVPageReuseObserver: AnyObject {
    func pagedKVPool(_ pool: PagedKVPool, isCached handle: PagedKVPageHandle) -> Bool
    func pagedKVPool(_ pool: PagedKVPool, willReuse handle: PagedKVPageHandle)
}
