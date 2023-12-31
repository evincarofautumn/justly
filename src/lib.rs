//! Collections that don’t need extra bounds-checking.
//!
//! # Overview
//!
//! **Justly** implements “justified” wrappers
//! for the standard Rust data structures
//! in [`std::collections`].
//! A *justified container* is one whose keys,
//! such as the [`usize`] indices into a [`Vec`],
//! keep track of their validity.
//! This has a few major consequences, listed below.
//!
//! ## Trustworthiness
//!
//! Justly is built on the idea of **trustworthy knowledge**:
//! [`call::Call`] represents a value called by a name,
//! which lets you encode *knowledge* about that named thing,
//! such as [`key::Key`], an index that you know is valid.
//! If you uphold the safety properties described in these APIs,
//! then such knowledge is called *trustworthy*,
//! and so you can rely on it to build safe abstractions.
//! Now unchecked indexing can be well and truly safe!
//!
//! ## No extra checks
//!
//! A “checked” method is one that performs bounds checking,
//! and an “unchecked” method is one that *unsafely* omits it.
//! Methods in Justly that *safely* omit bounds checks
//! are called **check-free**.
//! The Rust compiler attempts to elide checks
//! when it can prove that they will always succeed.
//! Justly implements check-free APIs
//! by giving you the tools to prove that checks can be skipped.
//!
//! It’s worth noting that some checks are strictly necessary!
//! For example, you *must* do a runtime test
//! to tell whether an arbitrary integer index is a valid key.
//! Justly only lets you avoid *extra* checks,
//! which are not strictly necessary,
//! yet the compiler cannot prove it.
//! But once you have gotten a typed key,
//! it proves that the test has already been done,
//! and doesn’t need to be repeated.
//!
//! Also, most indices that you get from the collection APIs
//! are already known to be valid keys,
//! so Justly merely adds that information in its wrapper APIs.
//!
//! ## No invalid indices
//!
//! When referring to elements in a collection,
//! Rust encourages using *indices*, relative to the collection,
//! instead of absolute pointers.
//! When a container is mutable,
//! mutating methods can reallocate the container’s storage
//! if its capacity is exceeded,
//! thereby invalidating all pointers to that storage.
//! So relative indexing makes it easier to uphold memory safety
//! compared to absolute addressing,
//! because indices automatically respond to *capacity* changes.
//!
//! However, relative indices
//! still don’t automatically handle changes in *size*:
//! any method that decreases the size of the container
//! must naturally invalidate any indices
//! referring to elements outside of its new bounds.
//!
//! Worse, since indices are just bare integers,
//! they come with no guarantees about the relationship
//! between the index and the value at that index.
//! For example, suppose we have a vector, `v`,
//! and we compute some indices into it, `i` and `j`.
//!
//! ```
//! let mut v = vec![1, 7, 16, 27, 40, 55];
//!
//! let i = v.iter().enumerate().position(|(k, &v)| k * 10 == v).unwrap();
//! let j = v.iter().enumerate().position(|(k, &v)| k * 11 == v).unwrap();
//!
//! println!("{}", v[i]);  // valid (40)
//! println!("{}", v[j]);  // valid (55)
//! ```
//!
//! If we `remove()` an element,
//! then this can easily break code
//! that makes assumptions about the indices:
//! `remove()` not only makes `j` inaccessible,
//! but also silently changes the meaning of `i`.
//! This is like a use-after-free error.
//!
//! ```should_panic
//! # let mut v = vec![1, 7, 16, 27, 40, 55];
//! # let i = 4;
//! # let j = 5;
//! v.remove(2);
//!
//! println!("{}", v[i]);  // logic error (55)
//! println!("{}", v[j]);  // runtime error (panic)
//! ```
//!
//! And of course if we `push()` an element,
//! the problem is compounded:
//! it silently makes `j` accessible again,
//! but with a different meaning as well.
//!
//! ```
//! # let mut v = vec![1, 7, 16, 27, 40, 55];
//! # let i = 4;
//! # let j = 5;
//! # v.remove(2);
//! v.push(68);
//!
//! println!("{}", v[j]);  // logic error (68)
//! ```
//!
//! So Justly wraps those *mutating* methods
//! that could invalidate the keys of a collection
//! in *consuming* methods that give back a modified object,
//! along with information about how the new keys
//! are related to the old ones.
//!
//! We do this with phantom type parameters,
//! so there is no runtime cost.
//!
//! # About this documentation
//!
//! Wrapper types and methods are annotated with tags
//! that describe their purpose, which are marked
//! with curly braces `{…}` for easier searching.
//!
//! ## {free}
//!
//! Marks methods that used to require bounds-checking
//! on the original type, but are now check-free.
//!
//! ## {just}
//!
//! On a container type, this says
//! that the type is a wrapper for a plain container
//! where some methods use typestate to add safety.
//!
//! On a method, it marks places
//! where we change the signature, typically either:
//!
//! * Taking ownership of `self`
//!   to replace a `mut` method on the wrapped type
//!   with one that tracks knowledge about the container
//!
//! * Adding wrappers such as [`key::Key`]
//!   to record some knowledge about the result
//!
//! ## {like}
//!
//! Links to the original wrapped thing.
//!
//! ## {safe}
//!
//! Marks things that used to be `unsafe` (usually *unchecked*)
//! but are now safe (implying check-freedom).
//!
//! # TODO
//!
//! ## Allow naming unsized things
//!
//! [`call::Call`] and [`link::Link`]
//! ask for `T: Sized`, but they don’t really need to,
//! since all the other fields are meant to be phantoms.
//!
//! ## Allocator support
//!
//! To support allocators,
//! there should be a type parameter on the wrappers,
//! such as `A = std::alloc::Global`.
//!
//! ## Implicit dereferencing
//!
//! We’re uneasy about whether to add implementations
//! of `Deref` and `DerefMut`
//! that would let a value be implicitly called by a name,
//! like these for `Vec`:
//!
//! ```ignore
//! impl<'name, T> Deref for Vec<'name, T> {
//!     type Target = Call<'name, vec::Vec<T>>;
//!     fn deref(&self) -> &Call<'name, vec::Vec<T>> {
//!         &self.same
//!     }
//! }
//!
//! impl<'name, T> DerefMut for Vec<'name, T> {
//!     fn deref_mut(
//!         &mut self,
//!     ) -> &mut Call<'name, vec::Vec<T>> {
//!         &mut self.same
//!     }
//! }
//! ```
//!
//! ## Tracking capacity
//!
//! At the time of this writing,
//! these APIs only track the validity of *indices*.
//! This means that if `std::collections` has a mutating method
//! that doesn’t change any indices,
//! we can just forward it and keep the same API,
//! which is likely better for usability.
//!
//! However, in principle
//! we could just as well track the *capacity* too.
//! This would reflect
//! the `std::collections` guarantees about allocation
//! at the type level,
//! making it possible in some circumstances
//! to statically guarantee that code will not allocate.
//! But it would come at the ergonomic cost
//! of using a different API
//! for functions that might change the capacity.
//!

#![feature(ptr_sub_ptr)]
#![feature(slice_split_at_unchecked)]
#![feature(slice_swap_unchecked)]
#![feature(slice_pattern)]
#![warn(missing_docs)]

pub mod name {

    //! Static, type-level names for runtime values.
    //!
    //! Names are represented using lifetime variables.
    //! For more on why, see [`mod@crate::call`].

    use core::marker::PhantomData;

    /// A name tag, which you can make for free.
    #[derive(Copy, Clone)]
    pub struct Name<'name>(PhantomData<&'name ()>);

    /// Make a name tag.
    pub const fn name<'name>() -> Name<'name> {
        Name(PhantomData)
    }
}

pub mod only {

    //! Helper for unique names.

    use super::name::{name, Name};

    /// Types whose values have unique names.
    ///
    /// Most likely you want [`crate::call::Call`]
    /// instead of this.
    ///
    /// # Safety
    ///
    /// `self` must uphold all trusted knowledge about `'name`.
    /// This means that `self` must be (or be the same as)
    /// the only bearer of this `'name`,
    /// so it must be immutable, or only ever used linearly.
    ///
    pub unsafe trait Only<'name> {
        /// Convenience method to get a name tag for `self`.
        fn name(&self) -> Name<'name> {
            name::<'name>()
        }
    }
}

pub mod call {

    //! `name ≡ value` relationships:
    //! *define* a unique static name for a dynamic value.
    //!
    //! To make a new name,
    //! we have to make up a new type-level constant.
    //! which is both statically known and unique.
    //!
    //! Often a “type constant” wants to be an existential type.
    //! However, `dyn Trait` types aren’t statically known,
    //! and `impl Trait` types aren’t necessarily unique.
    //! Happily, we can write rank-1 existential quantification
    //! using rank-2 universal quantification,
    //! which is allowed for lifetime variables.
    //!
    //! So we can write the name as a lifetime.
    //! That also lets us skip writing it sometimes,
    //! thanks to lifetime elision.
    //!

    use super::link::Link;
    use super::name::name;
    use super::only::Only;
    use core::ops::Deref;

    /// A value that’s called by a name.
    pub struct Call<'name, T> {
        link: Link<'name, T>,
    }

    unsafe impl<'name, T> Only<'name> for Call<'name, T> {}

    impl<'name, T> Call<'name, T> {
        /// Forgets its own name.
        pub fn into_owned(self) -> T {
            self.link.body
        }

        /// Changes the body *without* changing the name.
        ///
        /// # Safety
        ///
        /// Must uphold all trusted knowledge of `'name`.
        ///
        pub unsafe fn as_mut(&mut self) -> &mut T {
            &mut self.link.body
        }
    }

    impl<'name, T> Deref for Call<'name, T> {
        type Target = T;
        fn deref(&self) -> &T {
            &self.link.body
        }
    }

    /// Bestows a `'name` on something.
    ///
    /// Knowledge about the `'name`
    /// is now seen as knowledge about the named thing.
    ///
    /// As in “forge one’s own identity”, if you use it safely;
    /// or “commit forgery”, if you use it unsafely.
    ///
    /// If you just want a name and a value together,
    /// you likely need a plain [`crate::link::Link`].
    ///
    /// # Safety
    ///
    /// The thing must be the only one that bears this name.
    /// Otherwise, knowledge about the name might not be true.
    ///
    pub const unsafe fn forge<'name, T>(
        body: T,
    ) -> Call<'name, T> {
        Call {
            link: Link {
                body,
                name: name::<'name>(),
            },
        }
    }
}

pub mod called {

    //! Gives names to values.

    use super::call::{forge, Call};

    /// The set of types whose values can be called by a name.
    pub trait Called {
        /// Gives `self` a new name and evaluates `body`.
        ///
        /// Most often you will call this with a closure.
        ///
        /// ```ignore
        /// // TODO
        /// x.called(|x| {
        ///     // …
        /// })
        /// ```
        ///
        fn called<Out, Body>(self: Self, body: Body) -> Out
        where
            Body: for<'call> FnOnce(Call<'call, Self>) -> Out,
            Self: Sized;
    }

    /// Anything with a size can be called by a name.
    ///
    /// # TODO
    ///
    /// - [Allow naming unsized things][crate#allow-naming-unsized-things]
    ///
    impl<T> Called for T
    where
        T: Sized,
    {
        fn called<'here, Out, Body>(
            self: Self,
            body: Body,
        ) -> Out
        where
            Body: for<'call> FnOnce(Call<'call, Self>) -> Out,
            Self: Sized,
        {
            body(unsafe { forge(self) })
        }
    }
}

pub mod link {

    //! `name × value` relationships:
    //! *pair* a static name with a dynamic value.

    use super::name::{name, Name};

    /// A value, paired with a name
    /// that doesn’t have to be the name of the value itself.
    /// Often this is some kind of “reference”
    /// into the named parent structure,
    /// but it doesn’t need to be a Rust reference type,
    /// so to avoid potential confusion,
    /// we say there’s a “link” between them.
    ///
    /// # Safety
    ///
    /// It’s always *safe* to link to a name, but that means
    /// a link doesn’t necessarily say anything *trustworthy*
    /// about the relationship between `body` and `'name`.
    /// If you want that,
    /// you must make a newtype wrapper around it,
    /// or use one of the wrappers that Justly already offers,
    /// like [`crate::key::Key`].
    ///
    /// # TODO
    ///
    /// - [Allow naming unsized things][crate#allow-naming-unsized-things]
    ///
    #[derive(Copy, Clone)]
    pub struct Link<'name, T> {
        /// The value of the link.
        pub body: T,
        /// The name it is linked to.
        pub name: Name<'name>,
    }

    /// Pair a value with a name,
    /// where the value doesn’t lay claim to the name.
    pub const fn link<'name, T>(body: T) -> Link<'name, T> {
        Link {
            body,
            name: name::<'name>(),
        }
    }
}

pub mod prop {

    //! Common properties of collections.

    use super::name::Name;

    /// The property that a structure is sorted by `Ord`,
    /// that is, for indices `i` and `j`
    /// and their associated elements `n[i]` and `n[j]`,
    /// the association is monotonic:
    /// `i <= j` implies `n[i] <= n[j]`.

    pub struct Sorted<'name>(Name<'name>);
}

pub mod vec {

    //! [**{just}**](crate#just)
    //! [{like}](crate#like):[`mod@std::vec`]
    //! [{like}](crate#like):[`mod@std::slice`]
    //!
    //! `Vec<'name, T>` is a thin wrapper
    //! around a named vector, `Call<'name, std::vec::Vec<T>>`,
    //! which you can make
    //! using the [`crate::called::Called`] trait.
    //!
    //! As is typical for this library,
    //! the `Vec` wrapper replaces some methods
    //! with versions that track the state of the container
    //! to prove that unchecked indexing is safe,
    //! and offer some additional features enabled by this.
    //!
    //! The types `Slice` and `MutSlice`
    //! are similar thin wrappers for slices, and likewise
    //! `ConstPtr` and `MutPtr` for pointers.

    use super::call::{forge, Call};
    use super::key::Key;
    use super::link::{link, Link};
    use super::name::name;
    use super::only::Only;
    use super::proof::{assume, Eqv, Prf, Sub};
    use super::prop;
    use core::cmp::Ordering;
    use core::ops::Range;
    use core::slice;
    use core::slice::SlicePattern;
    use std::vec;

    /// A vector bestowed with a name.
    ///
    /// # Extensions
    ///
    /// - `contains_key`
    /// - `reverse_mut`
    ///
    /// # `Vec` support
    ///
    /// - [ ] [`std::vec::Vec::allocator()`]
    /// - [ ] [`std::vec::Vec::append()`]
    /// - [x] [`std::vec::Vec::as_mut_ptr()`]]
    /// - [x] [`std::vec::Vec::as_mut_slice()`]
    /// - [x] [`std::vec::Vec::as_ptr()`]
    /// - [x] [`std::vec::Vec::as_slice()`]
    /// - [x] [`std::vec::Vec::capacity()`]
    /// - [ ] [`std::vec::Vec::clear()`]
    /// - [ ] [`std::vec::Vec::dedup()`]
    /// - [ ] [`std::vec::Vec::dedup_by()`]
    /// - [ ] [`std::vec::Vec::dedup_by_key()`]
    /// - [ ] [`std::vec::Vec::drain()`]
    /// - [ ] [`std::vec::Vec::extend_from_slice()`]
    /// - [ ] [`std::vec::Vec::extend_from_within()`]
    /// - [x] [`std::vec::Vec::from_raw_parts()`]
    /// - [ ] [`std::vec::Vec::from_raw_parts_in()`]
    /// - [ ] [`std::vec::Vec::insert()`]
    /// - [ ] [`std::vec::Vec::into_boxed_slice()`]
    /// - [ ] [`std::vec::Vec::into_flattened()`]
    /// - [ ] [`std::vec::Vec::into_raw_parts()`]
    /// - [ ] [`std::vec::Vec::into_raw_parts_with_alloc()`]
    /// - [ ] [`std::vec::Vec::leak()`]
    /// - [x] [`std::vec::Vec::len()`]
    /// - [x] [`std::vec::Vec::new()`]
    /// - [ ] [`std::vec::Vec::new_in()`]
    /// - [ ] [`std::vec::Vec::pop()`]
    /// - [x] [`std::vec::Vec::push()`]
    /// - [ ] [`std::vec::Vec::push_within_capacity()`]
    /// - [ ] [`std::vec::Vec::remove()`]
    /// - [x] [`std::vec::Vec::reserve()`]
    /// - [x] [`std::vec::Vec::reserve_exact()`]
    /// - [ ] [`std::vec::Vec::resize()`]
    /// - [ ] [`std::vec::Vec::resize_with()`]
    /// - [ ] [`std::vec::Vec::retain()`]
    /// - [ ] [`std::vec::Vec::retain_mut()`]
    /// - [ ] [`std::vec::Vec::set_len()`]
    /// - [ ] [`std::vec::Vec::shrink_to()`]
    /// - [ ] [`std::vec::Vec::shrink_to_fit()`]
    /// - [ ] [`std::vec::Vec::spare_capacity_mut()`]
    /// - [ ] [`std::vec::Vec::splice()`]
    /// - [ ] [`std::vec::Vec::split_at_spare_mut()`]
    /// - [ ] [`std::vec::Vec::split_off()`]
    /// - [ ] [`std::vec::Vec::swap_remove()`]
    /// - [x] [`std::vec::Vec::truncate()`]
    /// - [x] [`std::vec::Vec::try_reserve_exact()`]
    /// - [x] [`std::vec::Vec::with_capacity()`]
    /// - [ ] [`std::vec::Vec::with_capacity_in()`]
    ///
    /// # `slice` support
    ///
    /// - [ ] [`slice::align_to()`]
    /// - [ ] [`slice::align_to_mut()`]
    /// - [ ] [`slice::array_chunks()`]
    /// - [ ] [`slice::array_chunks_mut()`]
    /// - [ ] [`slice::array_windows()`]
    /// - [ ] [`slice::as_chunks()`]
    /// - [ ] [`slice::as_chunks_mut()`]
    /// - [ ] [`slice::as_chunks_unchecked_mut()`]
    /// - [ ] [`slice::as_chunks_unchecked()`]
    /// - [x] [`slice::as_mut_ptr_range()`]
    /// - [x] [`slice::as_ptr_range()`]
    /// - [ ] [`slice::as_rchunks()`]
    /// - [ ] [`slice::as_rchunks_mut()`]
    /// - [ ] [`slice::as_simd()`]
    /// - [ ] [`slice::as_simd_mut()`]
    /// - [x] [`slice::binary_search()`]
    /// - [x] [`slice::binary_search_by()`]
    /// - [x] [`slice::binary_search_by_key()`]
    /// - [ ] [`slice::chunks()`]
    /// - [ ] [`slice::chunks_exact()`]
    /// - [ ] [`slice::chunks_exact_mut()`]
    /// - [ ] [`slice::chunks_mut()`]
    /// - [ ] [`slice::clone_from_slice()`]
    /// - [ ] [`slice::concat()`]
    /// - [x] [`slice::contains()`]
    /// - [ ] [`slice::copy_from_slice()`]
    /// - [ ] [`slice::copy_within()`]
    /// - [x] [`slice::ends_with()`]
    /// - [ ] [`slice::fill()`]
    /// - [ ] [`slice::fill_with()`]
    /// - [x] [`slice::get()`]
    /// - [ ] [`slice::get_many_mut()`]
    /// - [ ] [`slice::get_many_unchecked_mut()`]
    /// - [ ] [`slice::group_by()`]
    /// - [ ] [`slice::group_by_mut()`]
    /// - [ ] [`slice::is_empty()`]
    /// - [ ] [`slice::is_sorted()`]
    /// - [ ] [`slice::is_sorted_by_key()`]
    /// - [x] [`slice::iter()`]
    /// - [x] [`slice::iter_mut()`]
    /// - [ ] [`slice::join()`]
    /// - [ ] [`slice::partition_dedup()`]
    /// - [ ] [`slice::partition_dedup_by()`]
    /// - [ ] [`slice::partition_dedup_by_key()`]
    /// - [ ] [`slice::rchunks()`]
    /// - [ ] [`slice::rchunks_exact()`]
    /// - [ ] [`slice::rchunks_exact_mut()`]
    /// - [ ] [`slice::rchunks_mut()`]
    /// - [ ] [`slice::repeat()`]
    /// - [x] [`slice::reverse()`]
    /// - [ ] [`slice::rotate_left()`]
    /// - [ ] [`slice::rotate_right()`]
    /// - [ ] [`slice::rsplit()`]
    /// - [ ] [`slice::rsplit_array_mut()`]
    /// - [ ] [`slice::rsplit_array_ref()`]
    /// - [ ] [`slice::rsplit_mut()`]
    /// - [ ] [`slice::rsplitn()`]
    /// - [ ] [`slice::rsplitn_mut()`]
    /// - [ ] [`slice::select_nth_unstable()`]
    /// - [ ] [`slice::select_nth_unstable_by()`]
    /// - [ ] [`slice::select_nth_unstable_by_key()`]
    /// - [ ] [`slice::sort()`]
    /// - [ ] [`slice::sort_by()`]
    /// - [ ] [`slice::sort_by_key()`]
    /// - [ ] [`slice::sort_by_cached_key()`]
    /// - [x] [`slice::sort_unstable()`]
    /// - [ ] [`slice::sort_unstable_by()`]
    /// - [ ] [`slice::sort_unstable_by_key()`]
    /// - [ ] [`slice::split()`]
    /// - [ ] [`slice::split_array_mut()`]
    /// - [ ] [`slice::split_array_ref()`]
    /// - [x] [`slice::split_at()`]
    /// - [x] [`slice::split_at_mut()`]
    /// - [x] [`slice::split_at_mut_unchecked()`]
    /// - [x] [`slice::split_at_unchecked()`]
    /// - [ ] [`slice::split_inclusive()`]
    /// - [ ] [`slice::split_mut()`]
    /// - [ ] [`slice::splitn()`]
    /// - [ ] [`slice::splitn_mut()`]
    /// - [x] [`slice::starts_with()`]
    /// - [x] [`slice::strip_prefix()`]
    /// - [x] [`slice::strip_suffix()`]
    /// - [x] [`slice::swap()`]
    /// - [x] [`slice::swap_unchecked()`]
    /// - [ ] [`slice::swap_with_slice()`]
    /// - [ ] [`slice::take()`]
    /// - [ ] [`slice::take_first()`]
    /// - [ ] [`slice::take_first_mut()`]
    /// - [ ] [`slice::take_last()`]
    /// - [ ] [`slice::take_last_mut()`]
    /// - [ ] [`slice::take_mut()`]
    /// - [ ] [`slice::to_ascii_lowercase()`]
    /// - [ ] [`slice::to_ascii_uppercase()`]
    /// - [ ] [`slice::to_vec()`]
    /// - [ ] [`slice::to_vec_in()`]
    /// - [ ] [`slice::windows()`]
    ///
    /// # TODO
    ///
    /// - [Allocator support][crate#allocator-support]
    ///
    /// - [Implicit dereferencing][crate#implicit-dereferencing]
    ///

    pub struct Vec<'name, T> {
        own: Call<'name, vec::Vec<T>>,
    }

    /// A slice like `&[T]`
    /// but *known* to be of the named vector.
    /// [{like}](crate#like):[`prim@slice`]
    pub struct Slice<'name, 'vec, T> {
        #[allow(missing_docs)]
        pub own: Link<'name, &'vec [T]>,
    }

    impl<'name, 'vec, T> From<Link<'name, &'vec [T]>>
        for Slice<'name, 'vec, T>
    {
        fn from(own: Link<'name, &'vec [T]>) -> Self {
            Slice { own }
        }
    }

    /// A mutable slice like `&mut [T]`
    /// but *known* to be of the named vector.
    /// [{like}](crate#like):[`prim@slice`]
    pub struct MutSlice<'name, 'vec, T> {
        #[allow(missing_docs)]
        pub own: Link<'name, &'vec mut [T]>,
    }

    impl<'name, 'vec, T> From<Link<'name, &'vec mut [T]>>
        for MutSlice<'name, 'vec, T>
    {
        fn from(own: Link<'name, &'vec mut [T]>) -> Self {
            MutSlice { own }
        }
    }

    /// A pointer like `*const T`
    /// but *assumed* to be into the named vector.
    /// [{like}](crate#like):[`pointer`]
    pub struct ConstPtr<'name, T> {
        #[allow(missing_docs)]
        pub own: Link<'name, *const T>,
    }

    impl<'name, T> From<Link<'name, *const T>>
        for ConstPtr<'name, T>
    {
        fn from(own: Link<'name, *const T>) -> Self {
            ConstPtr { own }
        }
    }

    /// A mutable pointer like `*mut T`
    /// but *assumed* to be into the named vector.
    /// [{like}](crate#like):[`pointer`]
    pub struct MutPtr<'name, T> {
        #[allow(missing_docs)]
        pub own: Link<'name, *mut T>,
    }

    impl<'name, T> From<Link<'name, *mut T>> for MutPtr<'name, T> {
        fn from(own: Link<'name, *mut T>) -> Self {
            MutPtr { own }
        }
    }

    impl<'name, T> From<Call<'name, vec::Vec<T>>>
        for Vec<'name, T>
    {
        fn from(own: Call<'name, vec::Vec<T>>) -> Self {
            Vec { own }
        }
    }

    impl<'name, T> Vec<'name, T> {
        /// [{like}](crate#like):[`std::vec::Vec::new`]
        pub fn new() -> Vec<'name, T> {
            unsafe { Vec::from(forge(vec::Vec::new())) }
        }

        /// [{like}](crate#like):[`std::vec::Vec::with_capacity`]
        pub fn with_capacity(capacity: usize) -> Vec<'name, T> {
            unsafe {
                Vec::from(forge(vec::Vec::with_capacity(
                    capacity,
                )))
            }
        }

        /// [{like}](crate#like):[`std::vec::Vec::from_raw_parts`]
        pub unsafe fn from_raw_parts(
            ptr: *mut T,
            length: usize,
            capacity: usize,
        ) -> Vec<'name, T> {
            unsafe {
                Vec::from(forge(vec::Vec::from_raw_parts(
                    ptr, length, capacity,
                )))
            }
        }

        /// [{like}](crate#like):[`std::vec::Vec::capacity`]
        ///
        /// # TODO
        ///
        /// - [Tracking capacity](crate#tracking-capacity)

        pub fn capacity(&self) -> usize {
            self.own.capacity()
        }

        /// [{like}](crate#like):[`std::vec::Vec::reserve`]

        pub fn reserve(&mut self, additional: usize) -> () {
            unsafe { self.own.as_mut().reserve(additional) }
        }

        /// [{like}](crate#like):[`std::vec::Vec::reserve_exact`]
        ///
        /// # TODO
        ///
        /// - [Tracking capacity](crate#tracking-capacity)

        pub fn reserve_exact(
            &mut self,
            additional: usize,
        ) -> () {
            unsafe {
                self.own.as_mut().reserve_exact(additional)
            }
        }

        /// [{like}](crate#like):[`std::vec::Vec::try_reserve_exact`]

        pub fn try_reserve_exact(
            &mut self,
            additional: usize,
        ) -> Result<(), std::collections::TryReserveError>
        {
            unsafe {
                self.own.as_mut().try_reserve_exact(additional)
            }
        }
    } // impl Vec

    impl<'before_truncate, T> Vec<'before_truncate, T> {
        /// [**{just}**](crate#just)
        /// [{like}](crate#like):[`std::vec::Vec::truncate`]
        ///
        /// # Returns
        ///
        /// 0. Truncated vector
        /// 1. Proof that new keys are a subset of old keys
        /// 2. Partial mapping from old keys to new keys
        ///
        pub fn truncate<'after_truncate>(
            self,
            len: usize,
        ) -> (
            Vec<'after_truncate, T>,
            Prf<Sub<'after_truncate, 'before_truncate>>,
            impl Fn(
                Key<'before_truncate, usize>,
            )
                -> Option<Key<'after_truncate, usize>>,
        ) {
            let mut old = self.own.into_owned();
            old.truncate(len);
            (
                unsafe { Vec::from(forge(old)) },
                unsafe {
                    assume::<
                        Sub<'after_truncate, 'before_truncate>,
                    >()
                },
                move |key| {
                    if key.index() < len {
                        Some(unsafe {
                            Key::new(
                                key.index(),
                                name::<'after_truncate>(),
                            )
                        })
                    } else {
                        None
                    }
                },
            )
        }
    } // impl Vec

    impl<'name, T> Vec<'name, T> {
        /// [{like}](crate#like):[`std::vec::Vec::as_slice`]
        pub fn as_slice<'vec>(
            &'vec self,
        ) -> Slice<'name, 'vec, T>
        where
            'vec: 'name,
        {
            Slice::from(link(self.own.as_slice()))
        }

        /// [{like}](crate#like):[`std::vec::Vec::as_mut_slice`]
        pub fn as_mut_slice<'vec>(
            &'vec mut self,
        ) -> MutSlice<'name, 'vec, T>
        where
            'vec: 'name,
        {
            MutSlice::from(link(unsafe {
                self.own.as_mut().as_mut_slice()
            }))
        }

        /// [{like}](crate#like):[`std::vec::Vec::as_ptr`]
        pub unsafe fn as_ptr(&self) -> ConstPtr<'name, T> {
            ConstPtr::from(link(self.own.as_ptr()))
        }

        /// [{like}](crate#like):[`std::vec::Vec::as_mut_ptr`]
        pub unsafe fn as_mut_ptr(
            &mut self,
        ) -> MutPtr<'name, T> {
            MutPtr::from(link(self.own.as_mut().as_mut_ptr()))
        }

        /// [{like}](crate#like):[`slice::as_ptr_range()`]
        ///
        /// Since `Range` is half-open,
        /// the end point is *not* in the named vector.
        /// This doesn’t change safety,
        /// because `ConstPtr<'name, T>` doesn’t require
        /// that the pointer be in the named vector.
        pub unsafe fn as_ptr_range(
            &self,
        ) -> Range<ConstPtr<'name, T>> {
            let Range { start, end } = self.own.as_ptr_range();
            ConstPtr::from(link(start))
                ..ConstPtr::from(link(end))
        }

        /// [{like}](crate#like):[`slice::as_mut_ptr_range()`]
        ///
        /// # See Also
        ///
        /// - [`Vec::as_mut_ptr`].
        pub unsafe fn as_mut_ptr_range(
            &mut self,
        ) -> Range<MutPtr<'name, T>> {
            let Range { start, end } =
                self.own.as_mut().as_mut_ptr_range();
            MutPtr::from(link(start))..MutPtr::from(link(end))
        }

        /// [**{free}**](crate#free)
        /// [{like}](crate#like):[`slice::swap()`]
        pub fn swap(
            &mut self,
            a: Key<'name, usize>,
            b: Key<'name, usize>,
        ) {
            // Upholds `swap_unchecked`.
            debug_assert!(
                a.index() < self.len()
                    && b.index() < self.len()
            );
            unsafe {
                self.own
                    .as_mut()
                    .swap_unchecked(a.index(), b.index())
            }
        }

        /// [**{just}**](crate#just)
        /// {like}:[`std::vec::Vec::push()`]
        ///
        /// # Returns
        ///
        /// 0. Updated vector
        /// 1. Key of newly pushed element
        /// 2. Total mapping from old keys to new keys:
        ///    dom(in) ⊂ dom(out)
        ///
        pub fn push<'changed>(
            self,
            value: T,
        ) -> (
            Vec<'changed, T>,
            Key<'changed, usize>,
            impl Fn(Key<'name, usize>) -> Key<'changed, usize>,
        ) {
            let mut old = self.own.into_owned();
            let index = old.len();
            old.push(value);
            (
                unsafe { Vec::from(forge(old)) },
                unsafe { Key::new(index, name::<'changed>()) },
                |k| unsafe {
                    Key::new(k.index(), name::<'changed>())
                },
            )
        }

        /// [{like}](crate#like):[`std::vec::Vec::len`]
        ///
        /// # TODO
        ///
        /// We could annotate the result as evidence.
        pub fn len(&self) -> usize {
            self.own.len()
        }

        /// [**{safe}**](#safe)
        /// [{like}](crate#like):[`slice::swap_unchecked`]
        pub fn swap_unchecked(
            &mut self,
            a: Key<'name, usize>,
            b: Key<'name, usize>,
        ) {
            self.swap(a, b)
        }
    } // impl Vec

    impl<'before_reverse, T> Vec<'before_reverse, T> {
        /// [**{just}**](crate#just)
        /// [{like}](crate#like):[`slice::reverse()`]
        ///
        /// # Returns
        ///
        /// 0. Vector reversed by key–value pairs
        /// 1. Witness that all old keys are valid new keys:
        ///    dom(in) ≅ dom(out)
        ///
        pub fn reverse<'after_reverse>(
            self,
        ) -> (
            Vec<'after_reverse, T>,
            Prf<Eqv<'before_reverse, 'after_reverse>>,
            impl Fn(
                Key<'before_reverse, usize>,
            ) -> Key<'after_reverse, usize>,
        ) {
            let mut old = self.own.into_owned();
            old.reverse();
            let len = old.len();
            (
                unsafe { Vec::from(forge(old)) },
                unsafe {
                    assume::<Eqv<'before_reverse, 'after_reverse>>(
                    )
                },
                move |key| unsafe {
                    Key::new(
                        len - key.index() - 1,
                        name::<'after_reverse>(),
                    )
                },
            )
        }
    } // impl Vec

    impl<'name, T> Vec<'name, T> {
        /// [{like}](crate#like):[`slice::reverse()`]
        ///
        /// All keys stay valid, but their values are reversed.
        /// To invalidate the keys too, use [`Self::reverse()`].
        pub fn reverse_mut(&mut self) {
            unsafe { self.own.as_mut().reverse() }
        }

        /// [{like}](crate#like):[`slice::iter`]

        pub fn iter(&self) -> slice::Iter<'_, T> {
            self.own.iter()
        }

        /// [{like}](crate#like):[`slice::iter_mut`]

        pub fn iter_mut(&mut self) -> slice::IterMut<'_, T> {
            unsafe { self.own.as_mut().iter_mut() }
        }

        /// [**{free}**](crate#free)
        /// [{like}](crate#like):[`slice::split_at()`]

        pub fn split_at<'vec>(
            &'vec self,
            mid: Key<'name, usize>,
        ) -> (Slice<'name, 'vec, T>, Slice<'name, 'vec, T>)
        where
            'vec: 'name,
        {
            // Upholds `split_at_unchecked`.
            debug_assert!(mid.index() <= self.len());
            unsafe {
                let (left, right) =
                    self.own.split_at_unchecked(mid.index());
                (
                    Slice::from(link(left)),
                    Slice::from(link(right)),
                )
            }
        }

        /// [**{free}**](crate#free)
        /// [{like}](crate#like):[`slice::split_at_mut()`]

        pub fn split_at_mut<'vec>(
            &'vec mut self,
            mid: Key<'name, usize>,
        ) -> (MutSlice<'name, 'vec, T>, MutSlice<'name, 'vec, T>)
        where
            'vec: 'name,
        {
            // Upholds `split_at_mut_unchecked`:
            //
            // > The caller has to ensure that
            // > `0 <= mid <= self.len()`.
            //
            debug_assert!(mid.index() <= self.len());
            unsafe {
                let (left, right) = self
                    .own
                    .as_mut()
                    .split_at_mut_unchecked(mid.index());
                (
                    MutSlice::from(link(left)),
                    MutSlice::from(link(right)),
                )
            }
        }

        /// [**{safe}**](crate#safe)
        /// [{like}](crate#like):`slice::split_at_unchecked`

        pub fn split_at_unchecked<'vec>(
            &'vec self,
            mid: Key<'name, usize>,
        ) -> (Slice<'name, 'vec, T>, Slice<'name, 'vec, T>)
        where
            'vec: 'name,
        {
            self.split_at(mid)
        }

        /// [**{safe}**](crate#safe)
        /// [{like}](crate#like):[`slice::split_at_mut_unchecked`]

        pub fn split_at_mut_unchecked<'vec>(
            &'vec mut self,
            mid: Key<'name, usize>,
        ) -> (MutSlice<'name, 'vec, T>, MutSlice<'name, 'vec, T>)
        where
            'vec: 'name,
        {
            self.split_at_mut(mid)
        }

        /// [{like}](crate#like):[`slice::contains`]

        pub fn contains(&self, x: &T) -> bool
        where
            T: PartialEq<T>,
        {
            self.own.contains(x)
        }

        /// [{like}](crate#like):[`slice::starts_with`]

        pub fn starts_with(&self, needle: &[T]) -> bool
        where
            T: PartialEq<T>,
        {
            self.own.starts_with(needle)
        }

        /// [{like}](crate#like):[`slice::ends_with`]

        pub fn ends_with(&self, needle: &[T]) -> bool
        where
            T: PartialEq<T>,
        {
            self.own.ends_with(needle)
        }

        /// [{like}](crate#like):[`slice::strip_prefix`]

        pub fn strip_prefix<'vec, P>(
            &'vec self,
            prefix: &P,
        ) -> Option<Slice<'name, 'vec, T>>
        where
            P: SlicePattern<Item = T> + ?Sized,
            T: PartialEq<T>,
            'vec: 'name,
        {
            self.own
                .strip_prefix(prefix)
                .map(|slice| Slice::from(link(slice)))
        }

        /// [{like}](crate#like):[`slice::strip_suffix`]

        pub fn strip_suffix<'vec, P>(
            &'vec self,
            suffix: &P,
        ) -> Option<Slice<'name, 'vec, T>>
        where
            P: SlicePattern<Item = T> + ?Sized,
            T: PartialEq<T>,
            'vec: 'name,
        {
            self.own
                .strip_suffix(suffix)
                .map(|slice| Slice::from(link(slice)))
        }

        /// [**{just}**](crate#just)
        /// [{like}](crate#like):[`slice::binary_search`]
        ///
        /// The `Ok` index is guaranteed to be valid,
        /// so we can wrap it in a `Key`.
        /// The `Err` index may be `len()` (past the end).

        pub fn binary_search(
            &self,
            x: &T,
        ) -> Result<Key<'name, usize>, usize>
        where
            T: Ord,
        {
            self.own.binary_search(x).map(|index| unsafe {
                Key::new(index, name::<'name>())
            })
        }

        /// [{like}](crate#like):[`slice::binary_search_by`]

        pub fn binary_search_by<F>(
            &self,
            f: F,
        ) -> Result<Key<'name, usize>, usize>
        where
            F: FnMut(&T) -> Ordering,
        {
            self.own.binary_search_by(f).map(|index| unsafe {
                Key::new(index, name::<'name>())
            })
        }

        /// [{like}](crate#like):[`slice::binary_search_by_key`]

        pub fn binary_search_by_key<B, F>(
            &self,
            b: &B,
            f: F,
        ) -> Result<Key<'name, usize>, usize>
        where
            F: FnMut(&T) -> B,
            B: Ord,
        {
            self.own.binary_search_by_key(b, f).map(
                |index| unsafe {
                    Key::new(index, name::<'name>())
                },
            )
        }

        /// [{like}](crate#like):[`slice::sort_unstable`]

        pub fn sort_unstable<'changed>(
            self,
        ) -> (Vec<'changed, T>, Prf<prop::Sorted<'changed>>)
        where
            T: Ord,
        {
            let mut old = self.own.into_owned();
            old.sort_unstable();
            (unsafe { Vec::from(forge(old)) }, unsafe {
                assume::<prop::Sorted<'changed>>()
            })
        }

        /// [**{just}**](crate#just)
        /// [{like}](crate#like):[`std::collections::HashMap::contains_key()`]
        ///
        /// Instead of a mere yes-or-no `bool`,
        /// we return an `Option` of a known-valid `Key`.

        pub fn contains_key(
            &self,
            index: usize,
        ) -> Option<Key<'name, usize>> {
            self.own.get(index).map(|_val| unsafe {
                Key::new(index, name::<'name>())
            })
        }

        /// [**{free}**](crate#free)
        /// [{like}](crate#like):[`slice::get()`]

        pub fn get(&self, key: Key<'name, usize>) -> &T {
            unsafe { self.own.get_unchecked(key.index()) }
        }
    } // impl Vec

    impl<'name, 'vec, T> Slice<'name, 'vec, T> {
        /// Safely gets the starting index of a slice in its
        /// parent vector.

        pub fn index(
            &self,
            vec: &'vec Vec<'name, T>,
        ) -> Key<'name, usize> {
            unsafe {
                // Uphold `sub_ptr`.
                debug_assert!(vec
                    .own
                    .as_ptr_range()
                    .contains(&self.own.body.as_ptr()));
                Key::new(
                    self.own
                        .body
                        .as_ptr()
                        .sub_ptr(vec.own.as_ptr()),
                    vec.own.name(),
                )
            }
        }
    } // impl Slice

    impl<'name, 'vec, T> MutSlice<'name, 'vec, T> {
        /// Safely gets the starting index of a mutable slice in
        /// its parent vector.

        pub fn index(
            &mut self,
            vec: &'vec mut Vec<'name, T>,
        ) -> Key<'name, usize> {
            unsafe {
                // Uphold `sub_ptr`.
                debug_assert!(vec
                    .own
                    .as_ptr_range()
                    .contains(&self.own.body.as_ptr()));
                Key::new(
                    self.own
                        .body
                        .as_mut_ptr()
                        .sub_ptr(vec.own.as_mut().as_mut_ptr()),
                    vec.own.name(),
                )
            }
        }
    } // impl MutSlice
}

pub mod vec_deque {

    //! [{like}](crate#like):[`std::collections::vec_deque`]

    use super::call::Call;
    use std::collections::vec_deque;

    /// [{like}](crate#like):[`std::collections::VecDeque`]
    ///
    /// # TODO
    ///
    /// ## Missing `VecDeque` methods
    ///
    pub type VecDeque<'name, T> =
        Call<'name, vec_deque::VecDeque<T>>;
}

pub mod linked_list {

    //! [{like}](crate#like):[`std::collections::linked_list`]

    use super::call::Call;
    use std::collections::linked_list;

    /// [{like}](crate#like):[`std::collections::LinkedList`]
    ///
    /// # TODO
    ///
    /// ## Missing `LinkedList` methods
    ///
    pub type LinkedList<'name, T> =
        Call<'name, linked_list::LinkedList<T>>;
}

pub mod hash_map {

    //! [{like}](crate#like):[`std::collections::hash_map`]

    use super::call::Call;
    use std::collections::hash_map;

    /// [{like}](crate#like):[`std::collections::HashMap`]
    ///
    /// # TODO
    ///
    /// ## Missing `HashMap` methods
    ///
    pub type HashMap<'name, K, V, S = hash_map::RandomState> =
        Call<'name, hash_map::HashMap<K, V, S>>;
}

pub mod btree_map {

    //! [{like}](crate#like):[`std::collections::btree_map`]

    use super::call::Call;
    use std::collections::btree_map;

    /// [{like}](crate#like):[`std::collections::BTreeMap`]
    ///
    /// # TODO
    ///
    /// ## Missing `BTreeMap` methods
    ///
    pub type BTreeMap<'name, K, V> =
        Call<'name, btree_map::BTreeMap<K, V>>;
}

pub mod binary_heap {

    //! [{like}](crate#like):[`std::collections::binary_heap`]

    use super::call::Call;
    use std::collections::binary_heap;

    /// [{like}](crate#like):[`std::collections::BinaryHeap`]
    ///
    /// # TODO
    ///
    /// ## Missing `BinaryHeap` methods
    ///
    pub type BinaryHeap<'name, T> =
        Call<'name, binary_heap::BinaryHeap<T>>;
}

pub mod key {

    //! Indices known to be valid keys in collections.

    use super::link::Link;
    use super::name::Name;

    /// A `Key<'name, Index>` is an `Index`
    /// that is known to be a valid key in `'name`.
    #[derive(Copy, Clone)]
    pub struct Key<'name, Index> {
        own: Link<'name, Index>,
    }

    impl<'name, Index> Key<'name, Index> {
        /// Say that an index is a valid key in a collection.
        ///
        /// # Safety
        ///
        /// The index must really be a valid key.
        ///
        pub unsafe fn new(
            body: Index,
            name: Name<'name>,
        ) -> Self {
            Key {
                own: Link { body, name },
            }
        }

        /// Get the index of a key.
        pub fn index(&self) -> Index
        where
            Index: Copy,
        {
            self.own.body
        }
    }
}

pub mod proof {

    //! Static proofs.

    use super::name::Name;
    use core::marker::PhantomData;

    /// Proof of proposition `A`.
    pub struct Prf<A>(PhantomData<A>);

    /// False proposition. Provable by `contradiction`.
    pub enum False {}

    /// True proposition. Proof is `trivial`.
    pub enum True {}

    /// Propositional NOT.
    ///
    /// Provable with `not()`.
    ///
    pub struct Not<A>(PhantomData<A>);

    /// Propositional AND: provable by both proving `A` and
    /// proving `B`.
    pub struct And<A, B>(PhantomData<(A, B)>);

    /// Propositional OR, provable by either proving `A` or
    /// proving `B`; or set union.
    pub struct Or<A, B>(PhantomData<(A, B)>);

    /// Propositional implication, provable by proving `B` given
    /// a proof of `A`.
    pub struct Imp<A, B>(PhantomData<(A, B)>);

    /// Subset or equal.
    pub struct Sub<'s, 't>(Name<'s>, Name<'t>);

    /// Propositional equivalence, provable by proving that `A`
    /// if and only if `B`.
    pub struct Iff<A, B>(PhantomData<(A, B)>);

    /// Set equivalence.
    pub struct Eqv<'s, 't>(Name<'s>, Name<'t>);

    /// Unsafely assume a proposition. It’s best to use this
    /// with an explicit type argument (turbofish) to document
    /// the intention, like `assume::<Or<A, Not<A>>>()`.
    ///
    pub unsafe fn assume<A>() -> Prf<A> {
        axiom()
    }

    /// Privately define an axiom.
    fn axiom<A>() -> Prf<A> {
        Prf(PhantomData)
    }

    /// Implication introduction.
    pub fn lam<A, B>(_s: impl Fn(A) -> B) -> Prf<Imp<A, B>> {
        axiom()
    }

    /// Implication elimination.
    pub fn app<A, B>(
        _a_to_b: &Prf<Imp<A, B>>,
        _a: &Prf<A>,
    ) -> Prf<B> {
        axiom()
    }

    /// Conjunction introduction. Proves a conjunction by
    /// proving each conjunct.
    pub fn and<A, B>(_a: Prf<A>, _b: Prf<B>) -> Prf<And<A, B>> {
        axiom()
    }

    /// Conjunction elimination. Constructs a proof from a
    /// conjunction by extracting the *first* conjunct.
    pub fn fst<A, B>(_a_and_b: Prf<And<A, B>>) -> Prf<A> {
        axiom()
    }

    /// Conjunction elimination. Constructs a proof from a
    /// conjunction by extracting the *second* conjunct.
    pub fn snd<A, B>(_a_and_b: Prf<And<A, B>>) -> Prf<B> {
        axiom()
    }

    /// Disjunction elimination. Uses a disjunction by using
    /// each disjunct.
    pub fn or<A, B, C>(
        _a_or_b: &Prf<Or<A, B>>,
        _a_to_c: &Prf<Imp<A, C>>,
        _b_to_c: &Prf<Imp<B, C>>,
    ) -> Prf<C> {
        axiom()
    }

    /// Disjunction introduction. Proves a disjunction by
    /// proving the *first* disjunct.
    pub fn left<A, B>(_a: &Prf<A>) -> Prf<Or<A, B>> {
        axiom()
    }

    /// Disjunction introduction. Proves a disjunction by
    /// proving the *second* disjunct.
    pub fn right<A, B>(_b: &Prf<B>) -> Prf<Or<A, B>> {
        axiom()
    }

    /// Negation introduction.
    /// Refutes a proposition by proving
    /// that it implies contradiction.
    pub fn not<A>(_a_to_f: &Prf<Imp<A, False>>) -> Prf<Not<A>> {
        axiom()
    }

    /// Shows a contradiction.
    pub fn contradiction<A>(
        _a: &Prf<A>,
        _not_a: &Prf<Not<A>>,
    ) -> Prf<False> {
        axiom()
    }

    /// Proves the trivially true proposition,
    /// which holds by definition.
    pub fn trivial<A>() -> Prf<True> {
        axiom()
    }

    /// Proof by absurdity (*ex falso quodlibet*).
    ///
    pub fn impossible<A>(_f: &Prf<False>) -> Prf<A> {
        axiom()
    }

    /// Any proposition is equivalent to itself.
    ///
    pub fn iff_refl<A>() -> Prf<Iff<A, A>> {
        axiom()
    }

    /// Any proposition implies itself.
    ///
    pub fn imp_refl<A>() -> Prf<Imp<A, A>> {
        axiom()
    }

    /// Propositional equivalence is symmetric.
    ///
    pub fn iff_symm<A, B>(
        _a_iff_b: &Prf<Iff<A, B>>,
    ) -> Prf<Iff<B, A>> {
        axiom()
    }

    /// Relational composition.
    ///
    /// Law of transitivity for propositional equivalence.
    ///
    pub fn iff_trans<A, B, C>(
        _a_iff_b: &Prf<Iff<A, B>>,
        _b_iff_c: &Prf<Iff<B, C>>,
    ) -> Prf<Iff<A, C>> {
        axiom()
    }

    /// Functional composition.
    ///
    /// Law of transitivity for implication.
    ///
    pub fn imp_trans<A, B, C>(
        _a_imp_b: &Prf<Imp<A, B>>,
        _b_imp_c: &Prf<Imp<B, C>>,
    ) -> Prf<Imp<A, C>> {
        axiom()
    }

    /// Law of antisymmetry for implication.
    ///
    pub fn imp_antisymm<A, B>(
        _a_imp_b: &Prf<Imp<A, B>>,
        _b_imp_a: &Prf<Imp<B, A>>,
    ) -> Prf<Iff<A, B>> {
        axiom()
    }
}

#[cfg(test)]
mod tests {

    use super::called::Called;
    use super::key::Key;
    use super::vec::Vec;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn named_vec() {
        std::vec::Vec::new().called(|filenames| {
            let filenames = Vec::from(filenames);
            let (filenames, _, _) =
                filenames.push(String::from("taxes.txt"));
            let (filenames, _, _) =
                filenames.push(String::from("passwords.txt"));
            let passwords = filenames.contains_key(1).unwrap();
            assert_eq!(
                filenames.get(passwords),
                &"passwords.txt"
            );
            let (filenames, poems, upcast) =
                filenames.push(String::from("poems.txt"));
            {
                fn assert_type<'passwords, 'poems>(
                    _: &Key<'passwords, usize>,
                    _: &Key<'poems, usize>,
                    _: &impl Fn(
                        Key<'passwords, usize>,
                    )
                        -> Key<'poems, usize>,
                ) {
                }
                assert_type(&passwords, &poems, &upcast);
            }
            assert_eq!(filenames.get(poems), &"poems.txt");
            assert_eq!(
                filenames.get(upcast(passwords)),
                &"passwords.txt"
            );
        });
    }
}
