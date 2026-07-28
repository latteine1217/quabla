//! Experimental source-to-source AD transforms for a deliberately restricted
//! Rust expression subset.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, ExprClosure, Pat};

/// Builds a forward-mode differentiated scalar closure from a pure expression.
///
/// The input must be a one-argument closure written as `|x| expression`. The
/// generated closure takes `f64` and returns `nabla_core::Dual`, whose value
/// and derivative can be queried without building a runtime tape.
///
/// ```ignore
/// use nabla_core::forward_diff;
///
/// let f = forward_diff!(|x| x.sin() * x.exp() + 2.0 * x);
/// let result = f(0.7);
/// assert!(result.derivative().is_finite());
/// ```
///
/// This macro intentionally does not model arbitrary Rust control flow,
/// mutation, multiple differentiated inputs, or captured methods whose types
/// do not support `Dual` arithmetic.
#[proc_macro]
pub fn forward_diff(tokens: TokenStream) -> TokenStream {
    let closure = parse_macro_input!(tokens as ExprClosure);
    if closure.inputs.len() != 1 {
        return syn::Error::new_spanned(
            closure.inputs,
            "forward_diff! requires exactly one closure parameter",
        )
        .to_compile_error()
        .into();
    }
    let Some(Pat::Ident(input)) = closure.inputs.first() else {
        return syn::Error::new_spanned(
            closure.inputs,
            "forward_diff! requires an untyped identifier parameter such as |x|",
        )
        .to_compile_error()
        .into();
    };
    if input.by_ref.is_some() || input.mutability.is_some() || input.subpat.is_some() {
        return syn::Error::new_spanned(
            input,
            "forward_diff! parameter must be an immutable identifier",
        )
        .to_compile_error()
        .into();
    }

    let identifier = &input.ident;
    let body = &closure.body;
    quote! {
        |#identifier: f64| {
            let #identifier = ::nabla_core::Dual::variable(#identifier);
            #body
        }
    }
    .into()
}

/// Builds a scalar primal-and-gradient closure from a pure expression.
///
/// The input must be a closure with one or more untyped identifier parameters,
/// such as `|x, y| x * y + x.sin()`. Expansion evaluates the expression once
/// per parameter with that parameter seeded as active `Dual` and every other
/// parameter treated as a constant. The resulting closure returns
/// `nabla_core::ForwardGradient` in parameter order.
///
/// This is a forward-mode O(number of inputs) transform. It is a correctness
/// reference for small parameter sets, not a replacement for reverse-mode VJP.
#[proc_macro]
pub fn forward_gradient(tokens: TokenStream) -> TokenStream {
    let closure = parse_macro_input!(tokens as ExprClosure);
    if closure.inputs.is_empty() {
        return syn::Error::new_spanned(
            closure.inputs,
            "forward_gradient! requires at least one closure parameter",
        )
        .to_compile_error()
        .into();
    }
    let mut identifiers = Vec::with_capacity(closure.inputs.len());
    for pattern in &closure.inputs {
        let Pat::Ident(input) = pattern else {
            return syn::Error::new_spanned(
                pattern,
                "forward_gradient! requires untyped identifier parameters such as |x, y|",
            )
            .to_compile_error()
            .into();
        };
        if input.by_ref.is_some() || input.mutability.is_some() || input.subpat.is_some() {
            return syn::Error::new_spanned(
                input,
                "forward_gradient! parameters must be immutable identifiers",
            )
            .to_compile_error()
            .into();
        }
        identifiers.push(input.ident.clone());
    }

    let body = &closure.body;
    let result_identifiers = (0..identifiers.len())
        .map(|index| format_ident!("__nabla_forward_result_{index}"))
        .collect::<Vec<_>>();
    let evaluations = result_identifiers
        .iter()
        .enumerate()
        .map(|(active, result)| {
            let bindings = identifiers.iter().enumerate().map(|(index, identifier)| {
                if index == active {
                    quote! { let #identifier = ::nabla_core::Dual::variable(#identifier); }
                } else {
                    quote! { let #identifier = ::nabla_core::Dual::constant(#identifier); }
                }
            });
            quote! {
                let #result = {
                    #(#bindings)*
                    #body
                };
            }
        });
    let first_result = &result_identifiers[0];
    quote! {
        |#(#identifiers: f64),*| {
            #(#evaluations)*
            ::nabla_core::ForwardGradient::new(
                #first_result.value(),
                vec![#(#result_identifiers.derivative()),*],
            )
        }
    }
    .into()
}
