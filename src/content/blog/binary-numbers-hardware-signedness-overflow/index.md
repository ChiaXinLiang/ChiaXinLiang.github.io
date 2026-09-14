---
title: "Binary Numbers for Hardware: Bit Vectors, Signedness, and Overflow"
description: "Learn binary representation through explicit hardware contracts, small checked examples, and the path to a verified AI accelerator."
pubDate: "2026-09-14"
heroImage: "./section-overview.png"
series: "fpga-fundamentals"
code: "bits-1"
order: 2
topic: "Binary representation"
tags: ["FPGA", "Digital Hardware", "Fundamentals"]
---

## Overview

![Concept overview: Binary representation](./section-overview.png)

The overview separates three questions that hardware beginners often merge: which bits are present, how those bits are interpreted, and which width is retained after an operation. A wire vector carries a fixed number of binary digits. Signedness gives those digits a numeric interpretation. Arithmetic and assignment rules determine which resulting digits survive. Correct reasoning needs all three.

This lesson builds the number representation needed for later FPGA arithmetic. It uses small widths so every example can be checked by hand. The same principles apply to wider counters, addresses, tensor elements, and accumulators, but their valid ranges must be calculated for their own widths. A declaration such as four bits does not by itself say whether the value represents an unsigned count or a signed quantity.

Read the [FPGA introduction](/blog/what-is-an-fpga-programmable-hardware/) first if the distinction between logic and state is unfamiliar. You do not need a board to follow this lesson. The [foundations lab](/labs/fpga-fundamentals/foundations-lab.zip) checks decoding, extension, wrapped addition, and the matrix examples used later in the course. These are representation checks, rather than measurements of a particular FPGA implementation.

The figures use explicit bit positions and decimal values. When inspecting a result, first write its width and signedness, then decode it. This habit prevents a visually plausible binary answer from silently becoming a wrong mathematical value. It also gives a precise way to explain an overflow report.

## Deep dive

### A bit vector has positional weights

![Deep dive: A bit vector has positional weights](./deep-dive-component-01.png)

The first figure assigns a weight to each position in an unsigned six-bit vector. Starting from the least significant position, the weights are one, two, four, eight, sixteen, and thirty-two. A one selects its weight and a zero contributes nothing. The vector 101101 therefore represents thirty-two plus eight plus four plus one, which equals forty-five.

Indexing conventions should be stated rather than guessed from the drawing. For a declaration with positions five down to zero, bit five carries the weight thirty-two and bit zero carries the weight one. The textual vector is usually written most significant bit first. Reversing that order changes the value. If a protocol sends bits or bytes in a different order, that transport convention needs its own definition; it does not change the mathematical weight of a declared bit position.

An unsigned vector of width W represents integers from zero through two to the power W minus one. W is the number of retained bits. Six bits therefore provide sixty-four distinct patterns and a maximum value of sixty-three. Increasing the width adds representable patterns. It does not automatically increase the number of meaningful application values: a counter that visits only zero through nine still has ten legal states even if its register is wider.

A useful decoder can accumulate the selected weights in an ordinary unbounded reference integer. In the lab, the reference masks inputs to the declared width before interpreting them. This makes its contract explicit. A value outside the width cannot be accepted as if the hardware retained all its bits. For example, a value of sixty-four requires seven unsigned bits; keeping only six leaves the all-zero pattern.

Check the all-zero pattern, the all-one pattern, and a single one at each position before relying on a more elaborate arithmetic test. These cases expose reversed indexing and incorrect masks. Then enumerate all sixty-four patterns and verify that decoding and encoding agree within range.

Hardware interfaces benefit from the same discipline. An address, a packed flag field, and an arithmetic operand may all use six wires while carrying different meanings. Treating a collection of flags as one unsigned number might be convenient for display, but it does not establish that adding one is a meaningful operation. Representation is a necessary part of the interface specification, rather than a detail deferred until debugging.

### Two’s complement changes the leading weight

![Deep dive: Two’s complement changes the leading weight](./deep-dive-component-02.png)

Two's complement gives the leading bit a negative weight. For a signed six-bit vector, the weights are negative thirty-two, sixteen, eight, four, two, and one. The pattern 111101 sums to negative three. The identical wires interpreted as unsigned instead sum to sixty-one. The figure shows two interpretations of one pattern, not two different physical signals.

The signed range for width W extends from negative two to the power W minus one through positive two to the power W minus one minus one. Here the phrase W minus one is the exponent. For six bits, that range is negative thirty-two through thirty-one. It is asymmetric because zero uses one pattern and the leading negative weight allows one additional negative magnitude. A design must account for this asymmetry when taking absolute values or negating the most negative input.

One practical decoding rule first obtains the unsigned value. If the leading bit is zero, the signed value is unchanged. If that bit is one, subtract two to the power W. For 111101, sixty-one minus sixty-four equals negative three. The reference decoder uses this rule, which is independent of the later arithmetic DUT. It also handles the boundary pattern 100000 as negative thirty-two.

The representation does not announce itself on a waveform. A simulator display may show hexadecimal digits unless the viewer is configured to interpret them as signed. A log that prints the pattern D without its width or interpretation leaves important information missing. Four-bit 1101 is unsigned thirteen or signed negative three; an eight-bit pattern containing the same low digits may mean something else if its upper bits differ.

Mixed signed and unsigned expressions deserve particular attention in HDL. Declarations, literals, casts, and intermediate widths can influence the result. The [SystemVerilog author tutorial](https://systemverilog.dev/3.html) is useful alongside the language and tool documentation when introducing procedural hardware descriptions. In this course, the reference examples always specify interpretation rather than relying on implicit conversion.

Before comparing an output with a reference value, decide whether the comparison concerns the retained bit pattern or the represented integer. Both can be useful, but they answer different questions. A correct wrapped bit result may still be outside the application's allowed mathematical range. Recording both makes overflow and signedness failures easier to distinguish.

### Overflow is a width-dependent result

![Deep dive: Overflow is a width-dependent result](./deep-dive-component-03.png)

The overflow figure adds signed four-bit seven and one. The retained pattern is 1000, which represents negative eight under four-bit two's complement. The mathematical sum is positive eight, outside the signed range negative eight through seven. This mismatch is signed overflow. The hardware's low four bits can be exactly as specified even though they cannot represent the intended mathematical result.

Widening both operands before addition changes the contract. Five-bit 00111 plus 00001 produces 01000, which represents positive eight. The important step is retaining enough width in the computation, rather than only attaching a wider register after a previously truncated result. A source-level expression must be checked for its own sizing rules. The beginner lab uses explicit extension where the DUT should preserve a carry.

Carry-out and signed overflow are different conditions. Unsigned four-bit seven plus one fits in the range zero through fifteen. The resulting pattern 1000 is valid unsigned eight and no unsigned carry is required. The same addition overflows the signed four-bit range. Conversely, unsigned fifteen plus one wraps to zero with a carry, while the signed interpretation of 1111 plus 0001 is negative one plus one, which produces zero without signed overflow.

For equal-width two's-complement addition, a useful overflow check asks whether the operands have the same sign while the retained result has the opposite sign. This rule concerns addition under that representation; it should not be casually reused for subtraction or multiplication without deriving the relevant condition. The lab also compares the full reference sum against the permitted signed range, giving an independently understandable criterion.

Enumerate every signed four-bit operand pair to compare the two criteria. There are sixteen patterns for each operand and therefore two hundred fifty-six pairs. For each pair, retain the wrapped result and record whether the unbounded mathematical sum falls outside the representable range. Boundary examples alone can miss a mistake in a sign test.

The system response to overflow is another design decision. Wrapping, saturating, reporting an error, and using a wider accumulator are different behaviors. None follows automatically from the word signed. Later INT8 arithmetic uses wider accumulators because many products can contribute to one sum. This lesson establishes why a local result width must be justified before a complete accelerator's numerical accuracy can be discussed.

### Zero extension and sign extension differ

![Deep dive: Zero extension and sign extension differ](./deep-dive-component-04.png)

Extension adds upper bits while preserving a chosen interpretation. The left path starts with four-bit 1101 treated as unsigned thirteen. Adding four zeros produces eight-bit 00001101, still unsigned thirteen. The right path treats the initial pattern as signed negative three. Replicating its leading one produces 11111101, which remains signed negative three in eight-bit two's complement.

Sign extension follows from the positional weights, rather than a visual rule about filling empty space. Adding a new leading one introduces a larger negative weight while the old leading one becomes a positive weight. Their difference preserves the original negative contribution. Repeating this step for several added positions leaves the represented value unchanged. Positive signed values extend with zeros because their leading bit is zero.

Zero extension is appropriate for an unsigned count or address. Applying it to a negative two's-complement operand changes its meaning: four-bit 1101 becomes positive thirteen when the eight-bit leading bits are zero. Sign-extending a pattern intended as unsigned thirteen and later decoding it as signed instead produces negative three. The circuit must know which interpretation the interface intended.

Narrowing is a separate operation. Retaining the low four bits of a wider value does not guarantee preservation. Eight-bit positive thirteen narrowed to signed four bits becomes negative three. A safe conversion requires the source value to lie within the destination range, or it requires an explicit policy for an out-of-range input. Saturation should be implemented as a range check and selection, not described as an ordinary truncation.

Use three columns in conversion tests: original integer, original pattern with width and signedness, and destination pattern with width and signedness. Test the minimum and maximum values and values immediately outside the destination range. The lab enumerates extension over every four-bit pattern so both positive and negative cases are included.

This reasoning becomes practical when a narrow tensor element enters a wider multiplier or accumulator. Correctly extending an INT8 operand is part of preserving its value before arithmetic. The [INT8 arithmetic lesson](/blog/fpga-ai-number-1-int8-arithmetic/) applies the same principles to products, quantization, and sum bounds. First learn the representation here; then calculate the wider application's range instead of assuming that one extension example proves every numerical path.

## Conclusion

A binary answer is meaningful only when its width and interpretation are stated. Positional weights decode unsigned values, two's complement changes the leading weight, and extension preserves a value only under the intended signedness. Overflow reports that a mathematical result does not fit the retained representation; it is not interchangeable with carry-out.

Before writing an arithmetic interface, specify input widths, signedness, intermediate widths, destination range, and out-of-range behavior. Make the reference model use unbounded integers before applying the declared representation policy. Then compare the intended integer and the retained pattern separately when necessary. This is especially useful for tensor products and accumulations.

Continue with [combinational logic](/blog/combinational-logic-truth-tables-multiplexers/) to express complete functions over those vectors. The clocked-state and simulation lessons show how to retain results and check them across edges. Later, dot products and INT8 bounds connect these fundamentals to AI computation without skipping the representation contract.

### Sources

- [SystemVerilog procedural hardware descriptions](https://systemverilog.dev/3.html)
