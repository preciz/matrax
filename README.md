# Archived. Use the [Nx](https://github.com/elixir-nx/nx) library instead.

# Matrax

[![Build Status](https://travis-ci.org/preciz/matrax.svg?branch=master)](https://travis-ci.org/preciz/matrax)

A matrix library in pure Elixir based on `:atomics`.

Key features:
  - **concurrent accessibility**: atomics are mutable and can be accessed from multiple processes
  - **access path only transformations**: transformations like transpose change only the access path so the same matrix can be worked on in multiple states by different processes at the same time
  - **fast accessibility**: operations like `get/2` and `put/3` are very fast and based only on pure Elixir

```elixir
iex> matrax = Matrax.new(7, 4, seed_fun: fn _, {row, col} -> row + col end)
iex> matrax |> Matrax.to_list_of_lists()
[
    [0, 1, 2, 3],
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [3, 4, 5, 6],
    [4, 5, 6, 7],
    [5, 6, 7, 8],
    [6, 7, 8, 9]
]
```

## Installation

Add `matrax` to your list of dependencies in `mix.exs`:

**Note**: it requires OTP-21.2.1 or later.

```elixir
def deps do
  [
    {:matrax, "~> 0.3"}
  ]
end
```

## API summary
See [https://hexdocs.pm/matrax](https://hexdocs.pm/matrax) for full documentation.

* `Matrax.new/2` - Create new matrix.
* `Matrax.get/2` - Returns the integer at the given position.
* `Matrax.put/3` - Puts the given integer into the given position.
* `Matrax.add/3` - Adds the given increment to the value at given position atomically.
* `Matrax.add_get/3` - Adds the given increment to the value at given position atomically and returns result.
* `Matrax.sub/3` - Subtracts the given decrement to the value at given position atomically.
* `Matrax.sub_get/3` - Subtracts the given decrement to the value at given position atomically and returns result.
* `Matrax.exchange/3` - Exchanges the given integer at the given position atomically.
* `Matrax.compare_exchange/4` - Compares & exchanges the given integer at the given position atomically.
* `Matrax.index_to_position/2` - Converts the given atomics index to position tuple.
* `Matrax.position_to_index/2` - Converts the given position tuple to atomics index.
* `Matrax.min/1` - Returns smallest integer in matrix.
* `Matrax.max/1` - Returns largest integer in matrix.
* `Matrax.sum/1` - Returns sum of integers in matrix.
* `Matrax.member?/2` - Checks if value exists within matrix.
* `Matrax.apply/2` - Applies the given function to all elements of matrix.
* `Matrax.to_list/1` - Converts matrix to a flat list.
* `Matrax.to_list_of_lists/1` - Converts matrix to list of lists.
* `Matrax.row_to_list/2` - Converts row at given index of matrix to list.
* `Matrax.column_to_list/2` - Converts column at given index of matrix to list.
* `Matrax.copy/1` - Returns a copy of the matrix with a new atomics reference. Can be used to finish access path only modifications.
* `Matrax.transpose/1` - Transposes the given matrix. (access path modification only)
* `Matrax.flip_lr/1` - Flips the given matrix vertically. (access path modification only)
* `Matrax.flip_ud/1` - Flips the given matrix horizontally. (access path modification only)
* `Matrax.submatrix/3` - Returns a new submatrix. (access path modification only)
* `Matrax.reshape/3` Reshapes matrix. (access path modification only)
* `Matrax.diagonal/1` Returns diagonal of matrix. (access path modification only)
* `Matrax.row/2` Returns row of matrix. (access path modification only)
* `Matrax.column/2` Returns column of matrix. (access path modification only)
* `Matrax.argmin/1` Returns position with smallest value.
* `Matrax.argmax/1` Returns position with largest value.
* `Matrax.find/2` Returns position of value's first occurence or nil.
* `Matrax.identity/1` Creates new identity matrix.
* ...

## License

Matrax is [MIT licensed](LICENSE).
