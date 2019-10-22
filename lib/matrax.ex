defmodule Matrax do
  @moduledoc """
  Use `:atomics` as an M x N matrix.

  [Erlang atomics documentation](http://erlang.org/doc/man/atomics.html)

  ## Examples

      iex> matrax = Matrax.new(100, 100) # 100 x 100 matrix
      iex> matrax |> Matrax.put({0, 0}, 10) # add 10 to position {0, 0}
      iex> matrax |> Matrax.get({0, 0})
      10
      iex> matrax |> Matrax.add({0, 0}, 80)
      iex> matrax |> Matrax.get({0, 0})
      90

  ## Enumerable protocol

  `Matrax` implements the Enumerable protocol, so all Enum functions can be used:

      iex> matrax = Matrax.new(10, 10)
      iex> matrax |> Matrax.put({0, 0}, 8)
      iex> matrax |> Enum.max()
      8
      iex> matrax |> Enum.member?(7)
      false
  """

  @compile {:inline, position_to_index: 2, index_to_position: 2, count: 1}

  @keys [:atomics, :rows, :columns, :min, :max, :signed, :transposed]
  @enforce_keys @keys
  defstruct @keys

  @type t :: %__MODULE__{
          atomics: reference,
          rows: pos_integer,
          columns: pos_integer,
          min: integer,
          max: pos_integer,
          signed: boolean,
          transposed: boolean
        }

  @type position :: {row :: non_neg_integer, col :: non_neg_integer}

  @doc """
  Returns a new `%Matrax{}` struct.

  ## Options
    * `:seed_fun` - a function to seed all positions.  See `apply/2` for further information.
    * `:signed` - whether to have signed or unsigned 64bit integers

  ## Examples

       Matrax.new(10, 5) # 10 x 5 matrix
       Matrax.new(10, 5, signed: false) # unsigned integers
       Matrax.new(10, 5, seed_fun: fn _, {row, col} -> row * col end) # seed values
  """
  @spec new(pos_integer, pos_integer, list) :: t
  def new(rows, columns, options \\ []) when is_integer(rows) and is_integer(columns) do
    seed_fun = Keyword.get(options, :seed_fun, nil)
    signed = Keyword.get(options, :signed, true)

    atomics = :atomics.new(rows * columns, signed: signed)

    %{min: min, max: max} = :atomics.info(atomics)

    matrax = %Matrax{
      atomics: atomics,
      rows: rows,
      columns: columns,
      min: min,
      max: max,
      signed: signed,
      transposed: false
    }

    if seed_fun do
      Matrax.apply(matrax, seed_fun)
    end

    matrax
  end

  @doc """
  Returns a position tuple for the given atomics `index`.

  ## Examples

      iex> matrax = Matrax.new(10, 10)
      iex> Matrax.index_to_position(matrax, 10)
      {0, 9}
  """
  @spec index_to_position(t, pos_integer) :: position
  def index_to_position(%Matrax{columns: columns, transposed: false}, index)
      when is_integer(index) do
    index = index - 1

    {div(index, columns), rem(index, columns)}
  end

  def index_to_position(%Matrax{rows: rows, transposed: true}, index) when is_integer(index) do
    index = index - 1

    {rem(index, rows), div(index, rows)}
  end

  @doc """
  Returns atomics index corresponding to the position
  tuple in the given `%Matrax{}` struct.

  ## Examples

      iex> matrax = Matrax.new(10, 10)
      iex> matrax |> Matrax.position_to_index({1, 1})
      12
      iex> matrax |> Matrax.position_to_index({0, 4})
      5
  """
  @spec position_to_index(t, position) :: pos_integer
  def position_to_index(%Matrax{rows: rows, columns: columns, transposed: false}, {row, col})
      when row <= rows and col <= columns do
    row * columns + col + 1
  end

  def position_to_index(%Matrax{rows: rows, columns: columns, transposed: true}, {row, col})
      when row <= rows and col <= columns do
    col * rows + row + 1
  end

  @doc """
  Returns value at `position` from the given matrax.

  ## Examples

      iex> matrax = Matrax.new(10, 10, seed_fun: fn _ -> 3 end)
      iex> matrax |> Matrax.get({0, 5})
      3
  """
  @spec get(t, position) :: integer
  def get(%Matrax{atomics: atomics} = matrax, position) do
    index = position_to_index(matrax, position)

    :atomics.get(atomics, index)
  end

  @doc """
  Puts `value` into `matrax` at `position`.

  Returns `:ok`

  ## Examples

      iex> matrax = Matrax.new(10, 10)
      iex> matrax |> Matrax.put({1, 3}, 5)
      :ok
  """
  @spec put(t, position, integer) :: :ok
  def put(%Matrax{atomics: atomics} = matrax, position, value) when is_integer(value) do
    index = position_to_index(matrax, position)

    :atomics.put(atomics, index, value)
  end

  @doc """
  Adds `incr` to atomic at `position`.

  ## Examples

      iex> matrax = Matrax.new(10, 10)
      iex> matrax |> Matrax.add({0, 0}, 2)
      :ok
      iex> matrax |> Matrax.add({0, 0}, 2)
      :ok
      iex> matrax |> Matrax.get({0, 0})
      4
  """
  @spec add(t, position, integer) :: :ok
  def add(%Matrax{atomics: atomics} = matrax, position, incr) when is_integer(incr) do
    index = position_to_index(matrax, position)

    :atomics.add(atomics, index, incr)
  end

  @doc """
  Atomic addition and return of the result.

  Adds `incr` to atomic at `position` and returns result.

  ## Examples

      iex> matrax = Matrax.new(10, 10)
      iex> matrax |> Matrax.add_get({0, 0}, 2)
      2
      iex> matrax |> Matrax.add_get({0, 0}, 2)
      4
  """
  @spec add_get(t, position, integer) :: integer
  def add_get(%Matrax{atomics: atomics} = matrax, position, incr) when is_integer(incr) do
    index = position_to_index(matrax, position)

    :atomics.add_get(atomics, index, incr)
  end

  @doc """
  Subtracts `decr` from atomic at `position`.

  ## Examples

      iex> matrax = Matrax.new(10, 10)
      iex> matrax |> Matrax.sub({0, 0}, 1)
      :ok
      iex> matrax |> Matrax.sub({0, 0}, 1)
      :ok
      iex> matrax |> Matrax.get({0, 0})
      -2
  """
  @spec sub(t, position, integer) :: :ok
  def sub(%Matrax{atomics: atomics} = matrax, position, decr) when is_integer(decr) do
    index = position_to_index(matrax, position)

    :atomics.sub(atomics, index, decr)
  end

  @doc """
  Atomic subtraction and return of the result.

  Subtracts `decr` from atomic at `position` and returns result.

  ## Examples

      iex> matrax = Matrax.new(10, 10)
      iex> matrax |> Matrax.sub_get({0, 0}, 2)
      -2
      iex> matrax |> Matrax.sub_get({0, 0}, 2)
      -4
  """
  @spec sub_get(t, position, integer) :: integer
  def sub_get(%Matrax{atomics: atomics} = matrax, position, decr) when is_integer(decr) do
    index = position_to_index(matrax, position)

    :atomics.sub_get(atomics, index, decr)
  end

  @doc """
  Atomically compares the value at `position` with `expected` ,
  and if those are equal, sets value at `position` to `desired`.

  Returns :ok if `desired` was written.
  Returns the actual value at `position` if it does not equal to `desired`.

  ## Examples

      iex> matrax = Matrax.new(10, 10)
      iex> matrax |> Matrax.compare_exchange({0, 0}, 0, -10)
      :ok
      iex> matrax |> Matrax.compare_exchange({0, 0}, 3, 10)
      -10
  """
  @spec compare_exchange(t, position, integer, integer) :: :ok | integer
  def compare_exchange(%Matrax{atomics: atomics} = matrax, position, expected, desired)
      when is_integer(expected) and is_integer(desired) do
    index = position_to_index(matrax, position)

    :atomics.compare_exchange(atomics, index, expected, desired)
  end

  @doc """
  Atomically replaces value at `position` with `value` and
  returns the value it had before.

  ## Examples

      iex> matrax = Matrax.new(10, 10)
      iex> matrax |> Matrax.exchange({0, 0}, -10)
      0
      iex> matrax |> Matrax.exchange({0, 0}, -15)
      -10
  """
  @spec exchange(t, position, integer) :: integer
  def exchange(%Matrax{atomics: atomics} = matrax, position, value)
      when is_integer(value) do
    index = position_to_index(matrax, position)

    :atomics.exchange(atomics, index, value)
  end

  @doc """
  Returns count of values (rows * columns).

  ## Examples

      iex> matrax = Matrax.new(5, 5)
      iex> Matrax.count(matrax)
      25
      iex> Matrax.count(matrax) == :atomics.info(matrax.atomics).size
      true
  """
  @spec count(t) :: pos_integer
  def count(%Matrax{rows: rows, columns: columns}) do
    rows * columns
  end

  @doc """
  Returns smallest integer in `matrax`.

  ## Examples

      iex> matrax = Matrax.new(10, 10, seed_fun: fn _ -> 7 end)
      iex> matrax |> Matrax.min()
      7
  """
  @spec min(t) :: integer
  def min(%Matrax{atomics: atomics} = matrax) do
    last_index = count(matrax)

    do_min(atomics, last_index - 1, :atomics.get(atomics, last_index))
  end

  defp do_min(_, 0, acc), do: acc

  defp do_min(atomics, index, acc) do
    do_min(atomics, index - 1, Kernel.min(acc, :atomics.get(atomics, index)))
  end

  @doc """
  Returns largest integer in `matrax`.

  ## Examples

      iex> matrax = Matrax.new(10, 10, seed_fun: fn _, {row, col} -> row * col end)
      iex> matrax |> Matrax.max()
      81
  """
  @spec max(t) :: integer
  def max(%Matrax{atomics: atomics} = matrax) do
    last_index = count(matrax)

    do_max(atomics, last_index - 1, :atomics.get(atomics, last_index))
  end

  defp do_max(_, 0, acc), do: acc

  defp do_max(atomics, index, acc) do
    do_max(atomics, index - 1, Kernel.max(acc, :atomics.get(atomics, index)))
  end

  @doc """
  Returns sum of integers in `matrax`.

  ## Examples

      iex> matrax = Matrax.new(10, 10, seed_fun: fn _, {row, col} -> row * col end)
      iex> matrax |> Matrax.sum()
      2025
  """
  @spec sum(t) :: integer
  def sum(%Matrax{atomics: atomics} = matrax) do
    last_index = count(matrax)

    do_sum(atomics, last_index - 1, :atomics.get(atomics, last_index))
  end

  defp do_sum(_, 0, acc), do: acc

  defp do_sum(atomics, index, acc) do
    do_sum(atomics, index - 1, acc + :atomics.get(atomics, index))
  end

  @doc """
  Applies the given `fun` function to all elements of `matrax`.

  If arity of `fun` is 1 it receives the integer as single argument.
  If arity of `fun` is 2 it receives the integer as first and
  position tuple as the second argument.

  ## Examples
      iex> matrax = Matrax.new(10, 10)
      iex> matrax |> Matrax.apply(fn int -> int + 2 end)
      iex> matrax |> Matrax.get({0, 0})
      2
      iex> matrax = Matrax.new(10, 10)
      iex> matrax |> Matrax.apply(fn _int, {row, col} -> row * col end)
      iex> matrax |> Matrax.get({9, 9})
      81
  """
  @spec apply(t, (integer -> integer) | (integer, position -> integer)) :: :ok
  def apply(%Matrax{} = matrax, fun) when is_function(fun, 1) or is_function(fun, 2) do
    fun_arity = Function.info(fun)[:arity]

    do_apply(matrax, count(matrax), fun_arity, fun)
  end

  defp do_apply(_, 0, _, _), do: :ok

  defp do_apply(%Matrax{atomics: atomics} = matrax, index, fun_arity, fun) do
    value =
      case fun_arity do
        1 -> fun.(:atomics.get(atomics, index))
        2 -> fun.(:atomics.get(atomics, index), index_to_position(matrax, index))
      end

    :atomics.put(atomics, index, value)

    do_apply(matrax, index - 1, fun_arity, fun)
  end

  @doc """
  Converts `%Matrax{}` to a flat list.

  ## Examples

      iex> matrax = Matrax.new(3, 3, seed_fun: fn _, {row, col} -> row * col end)
      iex> Matrax.to_list(matrax)
      [0, 0, 0, 0, 1, 2, 0, 2, 4]
  """
  @spec to_list(t) :: list(integer)
  def to_list(%Matrax{rows: rows, columns: columns} = matrax) do
    for row <- 0..(rows - 1), col <- 0..(columns - 1) do
      get(matrax, {row, col})
    end
  end

  @doc """
  Converts `%Matrax{}` to list of lists.

  ## Examples

      iex> matrax = Matrax.new(5, 5, seed_fun: fn _, {row, col} -> row * col end)
      iex> Matrax.to_list_of_lists(matrax)
      [
        [0, 0, 0, 0, 0],
        [0, 1, 2, 3, 4],
        [0, 2, 4, 6, 8],
        [0, 3, 6, 9, 12],
        [0, 4, 8, 12, 16]
      ]
  """
  @spec to_list_of_lists(t) :: list(list(integer))
  def to_list_of_lists(%Matrax{rows: rows, columns: columns} = matrax) do
    for row <- 0..(rows - 1) do
      for col <- 0..(columns - 1) do
        get(matrax, {row, col})
      end
    end
  end

  @doc """
  Converts given row index of `%Matrax{}` to list.

  ## Examples
      iex> matrax = Matrax.new(5, 5, seed_fun: fn _, {row, col} -> row * col end)
      iex> matrax |> Matrax.row_to_list(2)
      [0, 2, 4, 6, 8]
  """
  @spec row_to_list(t, non_neg_integer) :: list(integer)
  def row_to_list(%Matrax{rows: rows} = matrax, row) when row in 0..(rows - 1) do
    for col <- 0..(matrax.columns - 1) do
      get(matrax, {row, col})
    end
  end

  @doc """
  Converts given column index of `%Matrax{}` to list.

  ## Examples
      iex> matrax = Matrax.new(5, 5, seed_fun: fn _, {row, col} -> row * col end)
      iex> matrax |> Matrax.column_to_list(2)
      [0, 2, 4, 6, 8]
  """
  @spec column_to_list(t, non_neg_integer) :: list(integer)
  def column_to_list(%Matrax{columns: columns} = matrax, col) when col in 0..(columns - 1) do
    for row <- 0..(matrax.rows - 1) do
      get(matrax, {row, col})
    end
  end

  @doc """
  Checks if `value` exists within `matrax`.

      iex> matrax = Matrax.new(5, 5, seed_fun: fn _, {row, col} -> row * col end)
      iex> matrax |> Matrax.member?(6)
      true
      iex> matrax |> Matrax.member?(100)
      false
  """
  @spec member?(t, integer) :: boolean
  def member?(%Matrax{min: min, max: max} = matrax, value) when is_integer(value) do
    case value do
      v when v < min or v > max ->
        false

      _else ->
        do_member?(matrax.atomics, count(matrax), value)
    end
  end

  defp do_member?(_, 0, _), do: false

  defp do_member?(atomics, index, integer) do
    case :atomics.get(atomics, index) do
      ^integer -> true
      _else -> do_member?(atomics, index - 1, integer)
    end
  end

  @doc """
  Returns a `%Matrax{}` struct with a new atomics reference
  and positional values identical to the given `matrax`.

  The returned copy is always `transposed: false` so this
  can be used to finish the access-path only transpose
  by the `transpose/1` function.


      iex> matrax = Matrax.new(10, 10)
      iex> matrax |> Matrax.put({0, 0}, -9)
      iex> matrax2 =  Matrax.copy(matrax)
      iex> Matrax.get(matrax2, {0, 0})
      -9
  """
  @spec copy(t) :: t
  def copy(%Matrax{} = matrax) do
    new_atomics = :atomics.new(count(matrax), signed: matrax.signed)

    matrax_copy = %Matrax{matrax | atomics: new_atomics, transposed: false}

    do_copy(matrax, matrax_copy, count(matrax))

    matrax_copy
  end

  defp do_copy(_, _, 0) do
    :done
  end

  defp do_copy(matrax, matrax_copy, index) do
    value = :atomics.get(matrax.atomics, index)

    put(matrax_copy, index_to_position(matrax, index), value)

    do_copy(matrax, matrax_copy, index - 1)
  end

  @doc """
  Only modifies the struct, it doesn't move or mutate data.

  Given `transposed: true` the access path to positions
  will be modified during execution in `position_to_index/2`.

  For a real transposed matrix with data modification
  you can first `transpose/1` then `copy/1`. `copy/1` creates a
  new `%Matrax{}` struct based on the transposed matrax.

  ## Examples

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
      iex> matrax |> Matrax.transpose() |> Matrax.to_list_of_lists()
      [
          [0, 1, 2, 3, 4, 5, 6],
          [1, 2, 3, 4, 5, 6, 7],
          [2, 3, 4, 5, 6, 7, 8],
          [3, 4, 5, 6, 7, 8, 9]
      ]
  """
  @spec transpose(t) :: t
  def transpose(%Matrax{} = matrax) do
    %Matrax{
      matrax
      | rows: matrax.columns,
        columns: matrax.rows,
        transposed: not matrax.transposed
    }
  end

  @doc """
  Returns a submatrix.

  Creates a new `:atomics` for the submatrix
  and copies values over.

  Ranges are inclusive.

  ## Examples

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
      iex> matrax |> Matrax.submatrix(0..3, 0..3) |> Matrax.to_list_of_lists()
      [
          [0, 1, 2, 3],
          [1, 2, 3, 4],
          [2, 3, 4, 5],
          [3, 4, 5, 6]
      ]
  """
  @spec submatrix(t, Range.t(), Range.t()) :: t | no_return
  def submatrix(
        %Matrax{rows: rows, columns: columns} = matrax,
        row_from..row_to,
        col_from..col_to
      )
      when row_from in 0..(rows - 1) and row_to in row_from..(rows - 1) and
             col_from in 0..(columns - 1) and col_to in col_from..(columns - 1) do
    submatrix_rows = row_to + 1 - row_from
    submatrix_columns = col_to + 1 - col_from

    submatrix_atomics = :atomics.new(submatrix_rows * submatrix_columns, signed: matrax.signed)

    submatrax = %Matrax{
      atomics: submatrix_atomics,
      rows: submatrix_rows,
      columns: submatrix_columns,
      min: matrax.min,
      max: matrax.max,
      signed: matrax.signed,
      transposed: false
    }

    Matrax.apply(submatrax, fn _, {row, col} -> get(matrax, {row + row_from, col + col_from}) end)

    submatrax
  end

  @doc """
  Reshapes `matrax` to the given `rows` & `cols`.

  ## Examples
      iex> matrax = Matrax.new(4, 3, seed_fun: fn _ -> 1 end)
      iex> matrax |> Matrax.to_list_of_lists()
      [
          [1, 1, 1],
          [1, 1, 1],
          [1, 1, 1],
          [1, 1, 1]
      ]
      iex> matrax |> Matrax.reshape(2, 6) |> Matrax.to_list_of_lists()
      [
          [1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1]
      ]
  """
  @spec reshape(t, pos_integer, pos_integer) :: t
  def reshape(%Matrax{rows: rows, columns: columns} = matrax, desired_rows, desired_columns)
      when rows * columns == desired_rows * desired_columns do
    %Matrax{matrax | rows: desired_rows, columns: desired_columns}
  end

  defimpl Enumerable do
    @moduledoc false

    alias Matrax

    def count(%Matrax{} = matrax) do
      {:ok, Matrax.count(matrax)}
    end

    def member?(%Matrax{} = matrax, int) do
      {:ok, Matrax.member?(matrax, int)}
    end

    def slice(%Matrax{atomics: atomics} = matrax) do
      {
        :ok,
        Matrax.count(matrax),
        fn start, length ->
          do_slice(atomics, start + 1, length)
        end
      }
    end

    defp do_slice(_, _, 0), do: []

    defp do_slice(atomics, index, length) do
      [:atomics.get(atomics, index) | do_slice(atomics, index + 1, length - 1)]
    end

    def reduce(%Matrax{atomics: atomics} = matrax, acc, fun) do
      do_reduce({atomics, 0, Matrax.count(matrax)}, acc, fun)
    end

    def do_reduce(_, {:halt, acc}, _fun), do: {:halted, acc}
    def do_reduce(tuple, {:suspend, acc}, fun), do: {:suspended, acc, &do_reduce(tuple, &1, fun)}
    def do_reduce({_, same, same}, {:cont, acc}, _fun), do: {:done, acc}

    def do_reduce({atomics, index, count}, {:cont, acc}, fun) do
      do_reduce(
        {atomics, index + 1, count},
        fun.(:atomics.get(atomics, index + 1), acc),
        fun
      )
    end
  end
end
