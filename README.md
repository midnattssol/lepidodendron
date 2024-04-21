# lepidodendron

`lepidodendron` is a ternary RISC virtual machine inspired by the [Trillium Architecture](https://homepage.cs.uiowa.edu/~dwjones/ternary/trillium.shtml) designed by Douglas W. Jones. `lepidodendron` comes with several tools:

- A small assembler which converts programs written in assembly to machine code.
- A `hexdump`-like utility for dumping the contents of a ternary (virtual) file.
- A virtual machine which can run programs from the machine code.

Note that `lepidodendron` is not stable and is under active development. No backwards compatibility is guaranteed.

## Example program

An example program looks like the following:

```
.data
    hi = "Hello, world!\0"

.code
    load r0, hi             ; Load the pointer into the string into the r0 register

    @putchar
        cmp r1, *r0         ; Compare the data at the current string pointer with 0
        triteq fl, 0        ; Tritwise equality with zero - (0->1, 1->0, -1->0)
        tritmul fl, 1       ; Zero all trits except the last one by tritwise multiplication
        loadnz pc, -1       ; Jump to -1 (exit) on encountering a null trite
        output *r0++        ; Write the result and increment the pointer register
        load pc, @putchar   ; Keep looping
```

which assembles to:

```
; Program memory
000000   109 840 dk9   cka 2v8 fzk   1va ––– –––   |·······––|

; RAM
000000   018 025 02c   02c 03f 00c   009 03t 03f   |Hello, wo|
000009   03k 02c 024   01g 000 000   000 000 000   |rld!·····|
00001k   000 000 000   000 000 000   000 000 000   |·········|
000010   000 000 000   000 000 000   000 000 000   |·········|
000019   000 000 000   000 000 000   000 000 000   |·········|
00002k   000 000 000   000 000 000   000 000 000   |·········|
000020   000 000 000   000 000 000   000 000 000   |·········|
000029   000 000 000   000 000 000   000 000 000   |·········|
00003k   000 000 000   000 000 000   000 000 000   |·········|
```

## Memory layout

`lepidodendron` has 2,187 (3^7) trytes of RAM. At 9 trits each, this makes for a total of 19,683 trits (corresponding to ~3.8 KB, an amount of memory comparable to the PDP-8). `lepidodendron` is based on the Harvard architecture, as it has separate storage and program memory. Program memory is currently theoretically unbounded.

`lepidodendron` uses the TERSCII encoding, also designed by Douglas W. Jones, to encode strings.

## Instructions

An instruction is a 9-trit word broken into the following:

- `XXX······`. Opcode.
- `···XX····`. Register.
- `·····XX··`. Address mode.
- `·······XX`. Address data.

Note that the address data may not necessarily be a memory address, but might also be an immediate value or an offset. See the Addressing modes header for more information.

### Operations

`lepidodendron` parses instructions in the format `opcode rg, val`, where `rg` is the register and `val` the read data (as explained in Addressing modes). Instructions are binary, unary, or nullary.

| Opcode   | Operation name | Description |
|:-------:|:--------------------|:------------|
| `store rg, val`    | Store               | Store the value `val` into memory at the location specified by `rg`. |
| `load rg, val`     | Load                | Perform side effects of memory access and load the value from memory at the location specified by `val` into register `rg`. |
| `loadnz rg, val`   | Load non-zero       | Perform side effects of memory access and, iff `fl` is zero, load the value from memory at the location specified by `val` into register `rg`. |
| `add rg, val`      | Add                 | Add the value `val` to the content of register `rg`. |
| `addc rg, val`     | Add with carry      | Add the value `val` and the carry flag to the content of register `rg`. |
| `mul rg, val`      | Multiply            | Multiply the content of register `rg` by the value `val`. |
| `div rg, val`      | Divide              | Divide the content of register `rg` by the value `val`. |
| `cmp rg, val`      | Compare             | Set the flag `fl` based on the comparison between the content of register `rg` and the value `val`. |
| `tsum rg, val`     | Tritwise sum        | Calculate the sum of the trits in `val`, storing the result in `rg`. |
| `rot rg, val`      | Rotate              | Rotate the content of register `rg` right by the number of bits specified by `val`. |
| `tritmul rg, val`  | Tritwise multiply   | Calculate the tritwise product of the content of register `rg` and the value `val`, storing the result in `rg`. |
| `triteq rg, val`   | Tritwise equality   | Check for tritwise equality between the content of register `rg` and the value `val`, storing the result in `rg`. |
| `input rg`    | Input               | Get a character in TERSCII encoding from input and store it in the register. |
| `output rg`   | Output              | Send a character in TERSCII encoding to the output. |
| `noop`     | No operation        | Do nothing. |

- **Note.** The value of `val` in unary instructions has no effect and will not be looked up. For example, the (malformed) instruction `input r0, *sp++` will not increase the stack pointer. While the assembler will raise a syntax error if this is used, the equivalent trytecode will not (which might result in executing garbage instructions in modes `nxt` and `x->disp`).

### Registers

| Name | Purpose |
|:--:|:----|
| `r0` | (General-purpose register) |
| `r1` | (General-purpose register) |
| `r2` | (General-purpose register) |
| `r3` | (General-purpose register) |
| `r4` | (General-purpose register) |
| `r5` | (General-purpose register) |
| `fl` (`r6`) | Arithmetic flags |
| `sp` (`r7`) | Stack pointer |
| `pc` (`r8`) | Program counter |
```

The alternative register names can be used, but will raise a warning. `sp` points to a location in memory and can be used to allocate values on the stack. The stack pointer starts out a the end of the statically allocated memory and grows from the top. `pc` stores the current program counter, starting at 0. Storing a negative number in `pc` will terminate the program.

#### `fl` — the flag register

`fl` is structured like `000_000_0cs`, where `s` is the sign (comparison) trit and `c` is the carry trit.

##### Sign trit

- s = 0 — equal
- s = + — greater
- s = - — less than

##### Carry trit

- c = 0 — no carry
- c = + — positive carry
- c = - — negative carry

### Addressing modes

| Example   | Addressing mode name | Function |
|:------- :|:--------------------|:-----|
| `rg`       | Register        | Read the register pointed to by the address data. |
| `300`     | Next tryte | Read from the next tryte in the code. |
| `2`      | Immediate | Read the address data as-is. |
| `*rg`      | Indirect | Lookup the memory at the location stored in `rg`. |
| `*++rg`    | Indirect preincrement   | Increment `rg` by 1 and look up the memory at the location stored in `rg`. |
| `*rg++`    | Indirect postincrement   | Look up the memory at the location stored in `rg` and increment `rg` by 1. |
| `*--rg`    | Indirect predecrement    | Decrement `rg` by 1 and look up the memory at the location stored in `rg`.|
| `*rg--`    | Indirect postdecrement    |Look up the memory at the location stored in `rg` and decrement `rg` by 1. |
| `rg->disp` | Indirect offset | Look up the memory at the location which is the sum of `rg` and the next tryte. |
```

The assembler automatically detects when a value is small enough to fit into the immediate value and uses it instead of the next tryte mode.

## Future plans

Some interesting things to add in the future would be the following:

- Instructions to explicitly switch "memory banks".
- Serialization of the system state.
- Basic file I/O for a virtual machine.
- Creating a small, Forth-like language which compiles to lepidodendron assembly.

## Justification

This project was done for multiple reasons. I wanted to figure out how ternary computers work, as well as challenging myself to write a useful command-line assembler which can provide understandable error messages. I think this project has also given me a better understanding of real-life RISC architectures and what challenges can occur when programming in assembly languages. I also wanted to reduce unneccessary interdependencies as far as possible, as well as use functional error-handling patterns. While I have written some VMs previously, this was an interesting experience and something I feel like I learnt quite a bit from.

## Sources

- [Tritwise computer](https://en.wikipedia.org/wiki/Ternary_computer)
- [Three-valued logic](https://en.wikipedia.org/wiki/Three-valued_logic)
- [Setun](https://en.wikipedia.org/wiki/Setun)
- [Balanced ternary](https://en.wikipedia.org/wiki/Balanced_ternary)
- Jones, W. Douglas. [Trillium Architecture](https://homepage.cs.uiowa.edu/~dwjones/ternary/trillium.shtml)
- Jones, W. Douglas. [TerSCII](https://homepage.cs.uiowa.edu/~dwjones/ternary/terscii.shtml)
- *Tritwise Computers: The Setun and the Setun 70.* Brusentsov, Nikolay Petrovich; Ramil Alvarez, José. Moscow State University
- Brousentsov N. P. et al. *Development of ternary computers at Moscow State University.*
