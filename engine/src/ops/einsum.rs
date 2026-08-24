// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Einstein summation: one notation for every product-and-sum over axes.
//!
//! `matmul` contracts the last axis of one operand with the second-to-last of
//! another, `bmm` does it with a batch in front, `dot` does it for vectors,
//! `trace` sums a diagonal, and an outer product has to be spelled as a
//! broadcast multiply. Those are all the same operation -- multiply, then sum
//! over the axes you did not ask to keep -- and the library had a separate name
//! for each arrangement of it and no name at all for most of them. `einsum` is
//! the arrangement written down instead of enumerated:
//!
//! ```text
//! "ij,jk->ik"          a matrix product
//! "ii->i"              the diagonal
//! "ii"                 the trace, because `i` appears twice and so is summed
//! "ij->ji"             a transpose
//! "i,j->ij"            an outer product
//! "...ij,...jk->...ik" a batched matrix product over any number of batch axes
//! "bhqd,bhkd->bhqk"    attention scores, which has no other name
//! ```
//!
//! It is not a convenience wrapper. The last line is the reason: a contraction
//! over four axes with two of them batched cannot be written with the operations
//! this library had, only worked around with a chain of permutes and reshapes
//! that the caller has to get right and that hides what is happening.
//!
//! What makes it worth having rather than merely possible is the *plan*. The
//! naive reading -- broadcast every operand against every axis, multiply, sum
//! what is left -- is correct and unusable: `"ij,jk->ik"` on `1000x1000` inputs
//! would build a `1000x1000x1000` intermediate to sum it back down again. So
//! operands are contracted a pair at a time, and each pair is permuted into
//! `(batch, left, contracted)` against `(batch, contracted, right)` and handed
//! to the same matrix multiply everything else in the library uses. The
//! intermediate is never larger than the result of that pair.
//!
//! Every step is an operation that already exists and already has a gradient --
//! `diagonal`, `sum`, `permute`, `reshape`, `matmul` -- so this contributes no
//! backward pass of its own. It is a plan, and the plan differentiates itself.

use crate::{
    error::{MinitensorError, Result},
    ops::{linalg, reduction, shape_ops},
    tensor::{Shape, Tensor},
};
use rustc_hash::{FxHashMap, FxHashSet};

/// Where the labels standing for ellipsis axes begin.
///
/// Above every letter, so the two never collide, and one label per axis the
/// ellipsis covers. Axis `j` counting *from the right* is always `ELLIPSIS + j`,
/// which is what makes an ellipsis over two axes line up with the last two of an
/// ellipsis over three: broadcasting aligns from the right, and giving the axes
/// names from the right makes that automatic rather than a special case.
const ELLIPSIS: usize = 256;

/// A parsed equation: one label per axis of each operand, and the labels the
/// result keeps, in order.
struct Plan {
    terms: Vec<Vec<usize>>,
    output: Vec<usize>,
}

/// Render a label the way the caller wrote it, for error messages.
fn label_name(label: usize) -> String {
    if label >= ELLIPSIS {
        format!("the ellipsis axis {} from the right", label - ELLIPSIS)
    } else {
        format!("subscript '{}'", label as u8 as char)
    }
}

/// Split an equation into its comma-separated inputs and its output, if it
/// names one.
///
/// Borrowed rather than owned throughout, and the whitespace is only stripped
/// into a new string when there is some to strip. An equation is parsed on
/// every call -- there is nowhere to cache it -- so the parse is the fixed cost
/// of the operation, and an allocation here is one paid by every caller
/// including the ones contracting four-by-four matrices.
fn split_equation(equation: &str) -> Result<(Vec<&str>, Option<&str>)> {
    let mut halves = equation.split("->");
    let lhs = halves.next().unwrap_or_default();
    let rhs = halves.next();
    if halves.next().is_some() {
        return Err(MinitensorError::invalid_argument(
            "einsum: the equation has more than one '->'",
        ));
    }
    Ok((lhs.split(',').collect(), rhs))
}

/// Which part of the equation an error is about.
///
/// A `usize` rather than a formatted string, because naming the term is only
/// needed when something is wrong and formatting it eagerly meant an allocation
/// per operand on every successful call.
#[derive(Clone, Copy)]
enum Term {
    Input(usize),
    Output,
}

impl std::fmt::Display for Term {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Term::Input(index) => write!(formatter, "subscript {index}"),
            Term::Output => write!(formatter, "the output"),
        }
    }
}

/// The labels a term names, with its ellipsis expanded to `rank` axes.
fn expand_term(term: &str, rank: usize, which: Term) -> Result<Vec<usize>> {
    let mut labels = Vec::new();
    let mut characters = term.chars().peekable();
    let mut seen_ellipsis = false;
    while let Some(character) = characters.next() {
        if character == '.' {
            // Two more dots have to follow, and only one ellipsis is allowed.
            if !(characters.next() == Some('.') && characters.next() == Some('.')) {
                return Err(MinitensorError::invalid_argument(format!(
                    "einsum: {which} contains a '.' that is not part of an ellipsis"
                )));
            }
            if seen_ellipsis {
                return Err(MinitensorError::invalid_argument(format!(
                    "einsum: {which} contains more than one ellipsis"
                )));
            }
            seen_ellipsis = true;
            // Left to right, so the leftmost axis is the furthest from the right.
            for offset in (0..rank).rev() {
                labels.push(ELLIPSIS + offset);
            }
        } else if character.is_ascii_alphabetic() {
            labels.push(character as usize);
        } else if !character.is_whitespace() {
            return Err(MinitensorError::invalid_argument(format!(
                "einsum: '{character}' is not a valid subscript in {which}"
            )));
        }
    }
    Ok(labels)
}

/// How many axes a term's ellipsis stands for, given the operand it describes.
fn ellipsis_rank(term: &str, ndim: usize, index: usize) -> Result<usize> {
    let named = term.chars().filter(|c| c.is_ascii_alphabetic()).count();
    if term.contains('.') {
        ndim.checked_sub(named).ok_or_else(|| {
            MinitensorError::invalid_argument(format!(
                "einsum: subscript {index} names {named} axes but the operand has only {ndim}"
            ))
        })
    } else if named == ndim {
        Ok(0)
    } else {
        Err(MinitensorError::invalid_argument(format!(
            "einsum: subscript {index} names {named} axes but the operand has {ndim}; \
             use an ellipsis to stand for the rest"
        )))
    }
}

fn parse(equation: &str, operands: &[Tensor]) -> Result<Plan> {
    let (raw_terms, raw_output) = split_equation(equation)?;
    if raw_terms.len() != operands.len() {
        return Err(MinitensorError::invalid_argument(format!(
            "einsum: the equation has {} subscripts but {} operands were given",
            raw_terms.len(),
            operands.len()
        )));
    }

    let mut ranks = Vec::with_capacity(raw_terms.len());
    for (index, (term, operand)) in raw_terms.iter().zip(operands).enumerate() {
        ranks.push(ellipsis_rank(term, operand.ndim(), index)?);
    }
    let covered = ranks.iter().copied().max().unwrap_or(0);

    let mut terms = Vec::with_capacity(raw_terms.len());
    for (index, term) in raw_terms.iter().enumerate() {
        terms.push(expand_term(term, ranks[index], Term::Input(index))?);
    }

    // How many terms each label appears in, counting a repeat within one term
    // once per occurrence -- an index used twice in the same operand is a
    // diagonal, and an index used in two operands is a contraction.
    let mut occurrences: FxHashMap<usize, usize> = FxHashMap::default();
    for labels in &terms {
        for &label in labels {
            *occurrences.entry(label).or_insert(0) += 1;
        }
    }

    let output = match raw_output {
        Some(text) => {
            let labels = expand_term(text, covered, Term::Output)?;
            let mut seen = FxHashSet::default();
            for &label in &labels {
                if !seen.insert(label) {
                    return Err(MinitensorError::invalid_argument(format!(
                        "einsum: the output names {} twice",
                        label_name(label)
                    )));
                }
                if !occurrences.contains_key(&label) {
                    return Err(MinitensorError::invalid_argument(format!(
                        "einsum: the output names {}, which no operand has",
                        label_name(label)
                    )));
                }
            }
            labels
        }
        None => {
            // Everything the ellipsis covers, then the subscripts used exactly
            // once, in the order their letters sort. That is NumPy's rule and
            // there is no better one to invent.
            let mut labels: Vec<usize> =
                (0..covered).rev().map(|offset| ELLIPSIS + offset).collect();
            let mut once: Vec<usize> = occurrences
                .iter()
                .filter(|(label, count)| **label < ELLIPSIS && **count == 1)
                .map(|(&label, _)| label)
                .collect();
            once.sort_unstable();
            labels.extend(once);
            labels
        }
    };

    Ok(Plan { terms, output })
}

/// Collapse a label that a single operand names twice onto its diagonal.
///
/// `"ii->i"` asks for the entries where both axes agree, which is what
/// `diagonal` returns -- and it returns them as a new last axis, so the label
/// list loses both occurrences and gains one at the end.
fn collapse_repeats(tensor: &mut Tensor, labels: &mut Vec<usize>) -> Result<()> {
    loop {
        let mut repeat = None;
        'search: for first in 0..labels.len() {
            for second in (first + 1)..labels.len() {
                if labels[first] == labels[second] {
                    repeat = Some((first, second));
                    break 'search;
                }
            }
        }
        let Some((first, second)) = repeat else {
            return Ok(());
        };
        *tensor = linalg::diagonal(tensor, 0, first as isize, second as isize)?;
        let label = labels[first];
        labels.remove(second);
        labels.remove(first);
        labels.push(label);
    }
}

/// The size each label stands for, and a complaint if two operands disagree.
///
/// A label of size one against a label of size `n` is the one disagreement that
/// is allowed: it broadcasts, as it does everywhere else in the library and as
/// it does in NumPy's `einsum`.
fn label_sizes(terms: &[Vec<usize>], operands: &[Tensor]) -> Result<FxHashMap<usize, usize>> {
    let mut sizes: FxHashMap<usize, usize> = FxHashMap::default();
    for (labels, operand) in terms.iter().zip(operands) {
        for (axis, &label) in labels.iter().enumerate() {
            let extent = operand.shape().dims()[axis];
            match sizes.get(&label) {
                None => {
                    sizes.insert(label, extent);
                }
                Some(&known) if known == extent || extent == 1 => {}
                Some(&1) => {
                    sizes.insert(label, extent);
                }
                Some(&known) => {
                    return Err(MinitensorError::invalid_argument(format!(
                        "einsum: {} is {known} on one operand and {extent} on another",
                        label_name(label)
                    )));
                }
            }
        }
    }
    Ok(sizes)
}

/// Grow every axis of size one whose label is larger elsewhere.
///
/// Done before any contraction so that the batch axes of a pair genuinely
/// match: a matrix multiply will not broadcast them for us, and expanding here
/// costs nothing because it is a stride of zero until something reads it.
fn broadcast_axes(
    tensor: &Tensor,
    labels: &[usize],
    sizes: &FxHashMap<usize, usize>,
) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    let wanted: Vec<isize> = labels
        .iter()
        .enumerate()
        .map(|(axis, label)| sizes.get(label).copied().unwrap_or(dims[axis]) as isize)
        .collect();
    if wanted
        .iter()
        .zip(dims)
        .all(|(&want, &have)| want as usize == have)
    {
        return Ok(tensor.clone());
    }
    tensor.expand(wanted)
}

/// Sum away the axes whose labels this operand alone carries and the result
/// does not want.
///
/// Done before contracting rather than after, because an axis summed early is
/// an axis the matrix multiply never has to carry.
fn sum_private_axes(
    tensor: &Tensor,
    labels: &mut Vec<usize>,
    keep: &FxHashSet<usize>,
) -> Result<Tensor> {
    let axes: Vec<isize> = labels
        .iter()
        .enumerate()
        .filter(|(_, label)| !keep.contains(label))
        .map(|(axis, _)| axis as isize)
        .collect();
    if axes.is_empty() {
        return Ok(tensor.clone());
    }
    labels.retain(|label| keep.contains(label));
    reduction::sum(tensor, Some(axes), false)
}

/// Move `tensor`'s axes into the order `wanted` names, then fold them into the
/// three groups a matrix multiply reads.
fn regroup(
    tensor: &Tensor,
    labels: &[usize],
    wanted: &[usize],
    groups: [usize; 3],
) -> Result<Tensor> {
    let order: Vec<isize> = wanted
        .iter()
        .map(|label| {
            labels
                .iter()
                .position(|candidate| candidate == label)
                .expect("every wanted label belongs to this operand") as isize
        })
        .collect();
    let permuted = tensor.permute(order)?.contiguous()?;
    permuted.reshape(Shape::new(groups.to_vec()))
}

/// Contract two operands into one.
///
/// Each label is one of four things, and which one decides where it goes:
/// present in both and wanted later, so it rides along as a batch axis; present
/// in both and wanted by nobody, so it is summed over -- that is the
/// contraction; or present in one alone, in which case it is a free axis of
/// whichever side has it. Arranged as `(batch, left, contracted)` against
/// `(batch, contracted, right)`, that is exactly a batched matrix product.
fn contract(
    left: &Tensor,
    left_labels: &[usize],
    right: &Tensor,
    right_labels: &[usize],
    keep: &FxHashSet<usize>,
    sizes: &FxHashMap<usize, usize>,
) -> Result<(Tensor, Vec<usize>)> {
    let in_right: FxHashSet<usize> = right_labels.iter().copied().collect();
    let in_left: FxHashSet<usize> = left_labels.iter().copied().collect();

    let mut batch = Vec::new();
    let mut contracted = Vec::new();
    let mut left_free = Vec::new();
    for &label in left_labels {
        if in_right.contains(&label) {
            if keep.contains(&label) {
                batch.push(label);
            } else {
                contracted.push(label);
            }
        } else {
            left_free.push(label);
        }
    }
    let right_free: Vec<usize> = right_labels
        .iter()
        .copied()
        .filter(|label| !in_left.contains(label))
        .collect();

    let extent = |labels: &[usize]| -> usize {
        labels
            .iter()
            .map(|label| sizes.get(label).copied().unwrap_or(1))
            .product()
    };
    let (batches, rows, inner, columns) = (
        extent(&batch),
        extent(&left_free),
        extent(&contracted),
        extent(&right_free),
    );

    let mut left_order = batch.clone();
    left_order.extend_from_slice(&left_free);
    left_order.extend_from_slice(&contracted);
    let mut right_order = batch.clone();
    right_order.extend_from_slice(&contracted);
    right_order.extend_from_slice(&right_free);

    let folded_left = regroup(left, left_labels, &left_order, [batches, rows, inner])?;
    let folded_right = regroup(right, right_labels, &right_order, [batches, inner, columns])?;
    let product = linalg::matmul(&folded_left, &folded_right)?;

    let mut labels = batch;
    labels.extend_from_slice(&left_free);
    labels.extend_from_slice(&right_free);
    let dims: Vec<usize> = labels
        .iter()
        .map(|label| sizes.get(label).copied().unwrap_or(1))
        .collect();
    Ok((product.reshape(Shape::new(dims))?, labels))
}

/// Evaluate an Einstein summation.
///
/// See the module documentation for the notation. Every step is an operation
/// that already carries a gradient, so this one does too, without a backward
/// pass of its own.
pub fn einsum(equation: &str, operands: &[Tensor]) -> Result<Tensor> {
    if operands.is_empty() {
        return Err(MinitensorError::invalid_argument(
            "einsum: at least one operand is required",
        ));
    }
    let plan = parse(equation, operands)?;

    // A subscript repeated inside one operand is a diagonal, and taking it
    // first leaves every remaining label naming exactly one axis of it.
    let mut terms = plan.terms;
    let mut tensors = operands.to_vec();
    for (tensor, labels) in tensors.iter_mut().zip(terms.iter_mut()) {
        collapse_repeats(tensor, labels)?;
    }

    let sizes = label_sizes(&terms, &tensors)?;
    for (tensor, labels) in tensors.iter_mut().zip(terms.iter()) {
        *tensor = broadcast_axes(tensor, labels, &sizes)?;
    }

    // An axis is worth keeping if the result wants it or another operand still
    // has to be contracted against it.
    let mut counts: FxHashMap<usize, usize> = FxHashMap::default();
    for labels in &terms {
        for &label in labels {
            *counts.entry(label).or_insert(0) += 1;
        }
    }
    let output_labels: FxHashSet<usize> = plan.output.iter().copied().collect();
    for index in 0..tensors.len() {
        let keep: FxHashSet<usize> = terms[index]
            .iter()
            .copied()
            .filter(|label| output_labels.contains(label) || counts[label] > 1)
            .collect();
        let mut labels = terms[index].clone();
        tensors[index] = sum_private_axes(&tensors[index], &mut labels, &keep)?;
        for label in &terms[index] {
            if !labels.contains(label) {
                *counts.get_mut(label).unwrap() -= 1;
            }
        }
        terms[index] = labels;
    }

    // Fold the operands together a pair at a time, left to right. What has to
    // survive a step is whatever the result wants plus whatever the operands
    // still waiting have.
    let (mut result, mut labels) = (tensors[0].clone(), terms[0].clone());
    for index in 1..tensors.len() {
        let mut keep = output_labels.clone();
        for later in &terms[index + 1..] {
            keep.extend(later.iter().copied());
        }
        let (folded, folded_labels) = contract(
            &result,
            &labels,
            &tensors[index],
            &terms[index],
            &keep,
            &sizes,
        )?;
        result = folded;
        labels = folded_labels;
    }

    // Anything the result does not want is summed, and what remains is put in
    // the order it asked for.
    let keep: FxHashSet<usize> = output_labels.clone();
    result = sum_private_axes(&result, &mut labels, &keep)?;
    let order: Vec<isize> = plan
        .output
        .iter()
        .map(|label| {
            labels
                .iter()
                .position(|candidate| candidate == label)
                .map(|axis| axis as isize)
                .ok_or_else(|| {
                    MinitensorError::internal_error(format!(
                        "einsum: {} was lost during the contraction",
                        label_name(*label)
                    ))
                })
        })
        .collect::<Result<Vec<_>>>()?;
    if order.len() == labels.len() && order.iter().enumerate().all(|(a, &b)| a as isize == b) {
        return Ok(result);
    }
    shape_ops::permute(&result, order)
}
