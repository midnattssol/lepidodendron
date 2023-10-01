# lepidodendron

A ternary RISC virtual machine.

Inspired by the [Trillium Architecture](https://homepage.cs.uiowa.edu/~dwjones/ternary/trillium.shtml) designed by Douglas W. Jones.

# Instructions — memory layout

(000)(00)(0000)

- opcode
- register
- address

## Addresses

address: (00)(00)
- mode
- register

### Addressing modes

x         register       
nxt       next tryte
10        immediate
*x        indirect       
*++x      indirect preincrement  
*x++      indirect postincrement  
*--x      indirect predecrement   
*x--      indirect postdecrement   
x->disp   indirect offset, offset is next tryte

# Operations

store     Store
load      Load
add       Add
sub       Subtract
addc      Add with carry
subb      Subtract with borrow
neg       Negation
shf       Shift
and       Minimum
or        Maximum
cons      Consensus
gull      Gullible
inw       Word in
outw      Word out
noop      Do nothing
cmp       Compare two numbers
swap      Swap memory

# Registers

r0        (General-purpose register)
r1        (General-purpose register)
r2        (General-purpose register)
r3        (General-purpose register)
r4        (General-purpose register)
r5        (General-purpose register)
fl        Arithmetic flags
sp        Stack pointer
pc        Program counter

## `fl`

`fl` is structured like `000_000_0cs`.

### Sign trit

- s = 0 — equal
- s = + — greater
- s = - — less than

### Carry trit

- c = 0 — no carry
- c = + — positive carry
- c = - — negative carry
