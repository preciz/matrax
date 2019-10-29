defmodule Matrax do
  @moduledoc """
  A matrix library in pure Elixir based on `atomics`.

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

  @compile {:inline,
            position_to_index: 2,
            do_position_to_index: 4,
            index_to_position: 2,
            do_index_to_position: 2,
            count: 1,
            put: 3,
            get: 2}

  @keys [:atomics, :rows, :columns, :min, :max, :signed, :changes]
  @enforce_keys @keys
  defstruct @keys

  @type t :: %__MODULE__{
          atomics: reference,
          rows: pos_integer,
          columns: pos_integer,
          min: integer,
          max: pos_integer,
          signed: boolean,
          changes: list
        }

  @type position :: {row :: non_neg_integer, col :: non_neg_integer}

  @doc """
  Converts a `list_of_lists` to a new `%Matrax{}` struct.

  ## Examples

       iex> matrax = %Matrax{rows: 2, columns: 3} = Matrax.new([[1,2,3], [4, 5, 6]])
       iex> matrax |> Matrax.to_list_of_lists
       [[1,2,3], [4, 5, 6]]
  """
  @spec new(list(list)) :: t
  def new(list_of_lists) do
    new(list_of_lists, [])
  end

  @doc """
  Converts a `list_of_lists` to a new `%Matrax{}` struct.

  ## Options
    * `:signed` - whether to have signed or unsigned 64bit integers

  ## Examples

       iex> matrax = %Matrax{rows: 2, columns: 3} = Matrax.new([[1,2,3], [4, 5, 6]], signed: false)
       iex> matrax |> Matrax.to_list_of_lists
       [[1,2,3], [4, 5, 6]]
       iex> matrax |> Matrax.count
       6
  """
  @spec new(list(list), list) :: t
  def new([first_list | _] = list_of_lists, options)
      when is_list(list_of_lists) and is_list(options) do
    rows = length(list_of_lists)
    columns = length(first_list)

    signed = Keyword.get(options, :signed, true)

    atomics = :atomics.new(rows * columns, signed: signed)

    list_of_lists
    |> List.flatten()
    |> Enum.reduce(1, fn value, index ->
      :atomics.put(atomics, index, value)

      index + 1
    end)

    %{min: min, max: max} = :atomics.info(atomics)

    %Matrax{
      atomics: atomics,
      rows: rows,
      columns: columns,
      min: min,
      max: max,
      signed: signed,
      changes: []
    }
  end

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
      changes: []
    }

    if seed_fun do
      Matrax.apply(matrax, seed_fun)
    end

    matrax
  end

  @doc """
  Create identity square matrix of given `size`.

  ## Examples

      iex> Matrax.identity(5) |> Matrax.to_list_of_lists
      [
          [1, 0, 0, 0, 0],
          [0, 1, 0, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 0, 1, 0],
          [0, 0, 0, 0, 1]
      ]
  """
  @spec identity(non_neg_integer) :: t
  def identity(size) when is_integer(size) and size > 0 do
    new(
      size,
      size,
      seed_fun: fn
        _, {same, same} -> 1
        _, {_, _} -> 0
      end
    )
  end

  @doc """
  Returns a position tuple for the given atomics `index`.

  Indices of atomix are 1 based.

  ## Examples

      iex> matrax = Matrax.new(10, 10)
      iex> Matrax.index_to_position(matrax, 1)
      {0, 0}
      iex> Matrax.index_to_position(matrax, 10)
      {0, 9}
  """
  @spec index_to_position(t, pos_integer) :: position
  def index_to_position(%Matrax{rows: rows, columns: columns}, index)
      when is_integer(index) and index <= rows * columns do
    do_index_to_position(columns, index)
  end

  defp do_index_to_position(columns, index) do
    index = index - 1

    {div(index, columns), rem(index, columns)}
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
  def position_to_index(%Matrax{rows: rows, columns: columns, changes: changes}, position) do
    do_position_to_index(changes, rows, columns, position)
  end

  defp do_position_to_index([], rows, columns, {row, col})
       when row >= 0 and row < rows and col >= 0 and col < columns do
    row * columns + col + 1
  end

  defp do_position_to_index([:transpose | changes_tl], rows, columns, {row, col}) do
    do_position_to_index(changes_tl, columns, rows, {col, row})
  end

  defp do_position_to_index(
         [{:reshape, {old_rows, old_columns}} | changes_tl],
         rows,
         columns,
         {row, col}
       ) do
    current_index = do_position_to_index([], rows, columns, {row, col})

    old_position = do_index_to_position(old_columns, current_index)

    do_position_to_index(changes_tl, old_rows, old_columns, old_position)
  end

  defp do_position_to_index(
         [
           {:submatrix, {old_rows, old_columns}, row_from.._row_to, col_from.._col_to}
           | changes_tl
         ],
         _,
         _,
         {row, col}
       ) do
    do_position_to_index(changes_tl, old_rows, old_columns, {row + row_from, col + col_from})
  end

  defp do_position_to_index(
         [{:diagonal, {old_rows, old_columns}} | changes_tl],
         1,
         _,
         {0, col}
       ) do
    do_position_to_index(changes_tl, old_rows, old_columns, {col, col})
  end

  defp do_position_to_index([:flip_lr | changes_tl], rows, columns, {row, col}) do
    do_position_to_index(changes_tl, rows, columns, {row, columns - 1 - col})
  end

  defp do_position_to_index([:flip_ud | changes_tl], rows, columns, {row, col}) do
    do_position_to_index(changes_tl, rows, columns, {rows - 1 - row, col})
  end

  defp do_position_to_index(
         [{:row, {old_rows, _old_columns}, row_index} | changes_tl],
         1,
         columns,
         {0, col}
       ) do
    do_position_to_index(changes_tl, old_rows, columns, {row_index, col})
  end

  defp do_position_to_index(
         [{:column, {_old_rows, old_columns}, col_index} | changes_tl],
         rows,
         1,
         {row, 0}
       ) do
    do_position_to_index(changes_tl, rows, old_columns, {row, col_index})
  end

  defp do_position_to_index(
         [{:drop_row, dropped_row_index} | changes_tl],
         rows,
         columns,
         {row, col}
       ) do
    row =
      if row >= dropped_row_index do
        row + 1
      else
        row
      end

    do_position_to_index(changes_tl, rows + 1, columns, {row, col})
  end

  defp do_position_to_index(
         [{:drop_column, dropped_column_index} | changes_tl],
         rows,
         columns,
         {row, col}
       ) do
    col =
      if col >= dropped_column_index do
        col + 1
      else
        col
      end

    do_position_to_index(changes_tl, rows, columns + 1, {row, col})
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

  Returns `:ok`.

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
  Adds a list of matrices to `matrax`.

  Size (rows, columns) of matrices must match.

  Returns `:ok`.

  ## Examples

      iex> matrax = Matrax.new(5, 5)
      iex> matrax7 = Matrax.new(5, 5, seed_fun: fn _ -> 7 end)
      iex> matrax |> Matrax.get({0, 0})
      0
      iex> matrax |> Matrax.add([matrax7, matrax7])
      iex> matrax |> Matrax.get({0, 0})
      14
      iex> matrax |> Matrax.add([matrax7])
      iex> matrax |> Matrax.get({0, 0})
      21
  """
  @spec add(t, list(t) | []) :: :ok
  def add(%Matrax{}, []) do
    :ok
  end

  def add(%Matrax{rows: rows, columns: columns} = matrax, [%Matrax{rows: rows, columns: columns} = head | tail]) do
    for row <- 0..(rows - 1), col <- 0..(columns - 1) do
      add(
        matrax,
        {row, col},
        get(head, {row, col})
      )
    end

    add(matrax, tail)
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
  Subtracts a list of matrices from `matrax`.

  Size (rows, columns) of matrices must match.

  Returns `:ok`.

  ## Examples

      iex> matrax = Matrax.new(5, 5)
      iex> matrax7 = Matrax.new(5, 5, seed_fun: fn _ -> 7 end)
      iex> matrax |> Matrax.get({0, 0})
      0
      iex> matrax |> Matrax.sub([matrax7, matrax7])
      iex> matrax |> Matrax.get({0, 0})
      -14
      iex> matrax |> Matrax.sub([matrax7])
      iex> matrax |> Matrax.get({0, 0})
      -21
  """
  @spec sub(t, list(t) | []) :: :ok
  def sub(%Matrax{}, []) do
    :ok
  end

  def sub(%Matrax{rows: rows, columns: columns} = matrax, [%Matrax{rows: rows, columns: columns} = head | tail]) do
    for row <- 0..(rows - 1), col <- 0..(columns - 1) do
      sub(
        matrax,
        {row, col},
        get(head, {row, col})
      )
    end

    sub(matrax, tail)
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
  def min(%Matrax{} = matrax) do
    {min_value, _position} = do_argmin(matrax)

    min_value
  end

  @doc """
  Returns largest integer in `matrax`.

  ## Examples

      iex> matrax = Matrax.new(10, 10, seed_fun: fn _, {row, col} -> row * col end)
      iex> matrax |> Matrax.max()
      81
      iex> Matrax.new(5, 5) |> Matrax.max()
      0
  """
  @spec max(t) :: integer
  def max(%Matrax{} = matrax) do
    {max_value, _position} = do_argmax(matrax)

    max_value
  end

  @doc """
  Returns sum of integers in `matrax`.

  ## Examples

      iex> matrax = Matrax.new(10, 10, seed_fun: fn _, {row, col} -> row * col end)
      iex> matrax |> Matrax.sum()
      2025
      iex> Matrax.new(5, 5, seed_fun: fn _ -> 1 end) |> Matrax.sum()
      25
  """
  @spec sum(t) :: integer
  def sum(%Matrax{} = matrax) do
    last_index = count(matrax)

    do_sum(matrax, last_index, 0)
  end

  defp do_sum(_, 0, acc), do: acc

  defp do_sum(matrax, index, acc) do
    position = index_to_position(matrax, index)

    do_sum(matrax, index - 1, acc + get(matrax, position))
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

  defp do_apply(%Matrax{} = matrax, index, fun_arity, fun) do
    position = index_to_position(matrax, index)

    value =
      case fun_arity do
        1 -> fun.(get(matrax, position))
        2 -> fun.(get(matrax, position), position)
      end

    put(matrax, position, value)

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
  def row_to_list(%Matrax{rows: rows, columns: columns} = matrax, row)
      when row in 0..(rows - 1) do
    for col <- 0..(columns - 1) do
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
  def column_to_list(%Matrax{rows: rows, columns: columns} = matrax, col)
      when col in 0..(columns - 1) do
    for row <- 0..(rows - 1) do
      get(matrax, {row, col})
    end
  end

  @doc """
  Only modifies the struct, it doesn't move or mutate data.

  Reduces matrix to only one row at given `row` index.

  After `row/2` the access path to positions will be
  modified during execution.

  If you want to get a new `:atomics` with mofified data
  use the `copy/1` function which applies the `:changes`.

  ## Examples

      iex> matrax = Matrax.new(5, 5, seed_fun: fn _, {row, _col} -> row end)
      iex> matrax |> Matrax.row(4) |> Matrax.to_list_of_lists
      [[4, 4, 4, 4, 4]]
  """
  @spec row(t, non_neg_integer) :: t
  def row(%Matrax{rows: rows, columns: columns, changes: changes} = matrax, row)
      when row in 0..(rows - 1) do
    %Matrax{matrax | rows: 1, changes: [{:row, {rows, columns}, row} | changes]}
  end

  @doc """
  Only modifies the struct, it doesn't move or mutate data.

  Reduces matrix to only one column at given `column` index.

  After `column/2` the access path to positions will be
  modified during execution.

  If you want to get a new `:atomics` with mofified data
  use the `copy/1` function which applies the `:changes`.

  ## Examples

      iex> matrax = Matrax.new(5, 5, seed_fun: fn _, {_row, col} -> col end)
      iex> matrax |> Matrax.column(4) |> Matrax.to_list_of_lists
      [[4], [4], [4], [4], [4]]
  """
  @spec column(t, non_neg_integer) :: t
  def column(%Matrax{rows: rows, columns: columns, changes: changes} = matrax, column)
      when column in 0..(columns - 1) do
    %Matrax{matrax | columns: 1, changes: [{:column, {rows, columns}, column} | changes]}
  end

  @doc """
  Checks if `value` exists within `matrax`.

  ## Examples

      iex> matrax = Matrax.new(5, 5, seed_fun: fn _, {row, col} -> row * col end)
      iex> matrax |> Matrax.member?(6)
      true
      iex> matrax |> Matrax.member?(100)
      false
  """
  @spec member?(t, integer) :: boolean
  def member?(%Matrax{} = matrax, value) when is_integer(value) do
    !!find(matrax, value)
  end

  @doc """
  Returns a `%Matrax{}` struct with a new atomics reference
  and positional values identical to the given `matrax`.

  The returned copy is always `changes: []` so this
  can be used to finish the access-path only changes
  by the `transpose/1`, `submatrix/3`, `reshape/3` functions.

  ## Examples

      iex> matrax = Matrax.new(10, 10)
      iex> matrax |> Matrax.put({0, 0}, -9)
      iex> matrax2 =  Matrax.copy(matrax)
      iex> Matrax.get(matrax2, {0, 0})
      -9
  """
  @spec copy(t) :: t
  def copy(%Matrax{atomics: atomics, changes: changes, signed: signed, columns: columns} = matrax) do
    size = count(matrax)

    new_atomics_ref = :atomics.new(size, signed: signed)

    case changes do
      [] -> do_copy(size, atomics, new_atomics_ref)
      [_ | _] -> do_copy(size, matrax, new_atomics_ref, columns)
    end

    %Matrax{matrax | atomics: new_atomics_ref, changes: []}
  end

  defp do_copy(0, _, _) do
    :done
  end

  defp do_copy(index, atomics, new_atomics_ref) do
    :atomics.put(new_atomics_ref, index, :atomics.get(atomics, index))

    do_copy(index - 1, atomics, new_atomics_ref)
  end

  defp do_copy(0, _, _, _) do
    :done
  end

  defp do_copy(index, matrax, new_atomics_ref, columns) do
    value = get(matrax, {div(index - 1, columns), rem(index - 1, columns)})

    :atomics.put(new_atomics_ref, index, value)

    do_copy(index - 1, matrax, new_atomics_ref, columns)
  end

  @doc """
  Only modifies the struct, it doesn't move or mutate data.

  After `transpose/1` the access path to positions
  will be modified during execution.

  If you want to get a new `:atomics` with mofified data
  use the `copy/1` function which applies the `:changes`.

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
  def transpose(
        %Matrax{rows: rows, columns: columns, changes: [:transpose | changes_tl]} = matrax
      ) do
    %Matrax{
      matrax
      | rows: columns,
        columns: rows,
        changes: changes_tl
    }
  end

  def transpose(%Matrax{rows: rows, columns: columns, changes: changes} = matrax) do
    %Matrax{
      matrax
      | rows: columns,
        columns: rows,
        changes: [:transpose | changes]
    }
  end

  @doc """
  Only modifies the struct, it doesn't move or mutate data.

  After `diagonal/1` the access path to positions will be
  modified during execution.

  If you want to get a new `:atomics` with mofified data
  use the `copy/1` function which applies the `:changes`.

  ## Examples

      iex> matrax = Matrax.identity(5)
      iex> matrax |> Matrax.diagonal() |> Matrax.to_list_of_lists
      [[1, 1, 1, 1, 1]]
  """
  @spec diagonal(t) :: t
  def diagonal(%Matrax{rows: rows, columns: columns, changes: changes} = matrax) do
    %Matrax{
      matrax
      | rows: 1,
        columns: rows,
        changes: [{:diagonal, {rows, columns}} | changes]
    }
  end

  @doc """
  Only modifies the struct, it doesn't move or mutate data.

  Ranges are inclusive.

  After `submatrix/3` the access path to positions will be
  modified during execution.

  If you want to get a new `:atomics` with mofified data
  use the `copy/1` function which applies the `:changes`.

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
      iex> matrax |> Matrax.submatrix(5..6, 1..3) |> Matrax.to_list_of_lists()
      [
          [6, 7, 8],
          [7, 8, 9]
      ]
  """
  @spec submatrix(t, Range.t(), Range.t()) :: t
  def submatrix(
        %Matrax{rows: rows, columns: columns, changes: changes} = matrax,
        row_from..row_to = row_range,
        col_from..col_to = col_range
      )
      when row_from in 0..(rows - 1) and row_to in row_from..(rows - 1) and
             col_from in 0..(columns - 1) and col_to in col_from..(columns - 1) do
    submatrix_rows = row_to + 1 - row_from
    submatrix_columns = col_to + 1 - col_from

    %Matrax{
      matrax
      | rows: submatrix_rows,
        columns: submatrix_columns,
        changes: [{:submatrix, {rows, columns}, row_range, col_range} | changes]
    }
  end

  @doc """
  Returns position tuple of biggest value.

  ## Examples

      iex> matrax = Matrax.new(5, 5, seed_fun: fn _, {row, col} -> row * col end)
      iex> matrax |> Matrax.argmax()
      {4, 4}
      iex> matrax = Matrax.new(5, 5) # all zeros
      iex> matrax |> Matrax.argmax()
      {0, 0}
  """
  @spec argmax(t) :: integer
  def argmax(%Matrax{} = matrax) do
    {_, position} = do_argmax(matrax)

    position
  end

  defp do_argmax(matrax) do
    acc = {get(matrax, {0, 0}), {0, 0}}

    do_argmax(matrax, 1, count(matrax), acc)
  end

  defp do_argmax(_, same, same, acc) do
    acc
  end

  defp do_argmax(matrax, index, size, {acc_value, _acc_position} = acc) do
    next_index = index + 1

    position = index_to_position(matrax, next_index)

    value_at_index = get(matrax, position)

    do_argmax(
      matrax,
      next_index,
      size,
      case Kernel.max(acc_value, value_at_index) do
        ^acc_value -> acc
        _else -> {value_at_index, position}
      end
    )
  end

  @doc """
  Returns position tuple of smallest value.

  ## Examples

      iex> matrax = Matrax.new(5, 5, seed_fun: fn _, {row, col} -> row * col end)
      iex> matrax |> Matrax.argmin()
      {0, 0}
      iex> matrax = Matrax.new(5, 5, seed_fun: fn _, {row, col} -> -(row * col) end)
      iex> matrax |> Matrax.argmin()
      {4, 4}
  """
  @spec argmin(t) :: integer
  def argmin(%Matrax{} = matrax) do
    {_, position} = do_argmin(matrax)

    position
  end

  defp do_argmin(matrax) do
    acc = {get(matrax, {0, 0}), {0, 0}}

    do_argmin(matrax, 1, count(matrax), acc)
  end

  defp do_argmin(_, same, same, acc) do
    acc
  end

  defp do_argmin(matrax, index, size, {acc_value, _acc_position} = acc) do
    next_index = index + 1

    position = index_to_position(matrax, next_index)

    value_at_index = get(matrax, position)

    do_argmin(
      matrax,
      next_index,
      size,
      case Kernel.min(acc_value, value_at_index) do
        ^acc_value -> acc
        _else -> {value_at_index, position}
      end
    )
  end

  @doc """
  Reshapes `matrax` to the given `rows` & `cols`.

  After `reshape/3` the access path to positions will be
  modified during execution.

  If you want to get a new `:atomics` with mofified data
  use the `copy/1` function which applies the `:changes`.

  ## Examples

      iex> matrax = Matrax.new(4, 3, seed_fun: fn _, {_row, col} -> col end)
      iex> matrax |> Matrax.to_list_of_lists()
      [
          [0, 1, 2],
          [0, 1, 2],
          [0, 1, 2],
          [0, 1, 2]
      ]
      iex> matrax |> Matrax.reshape(2, 6) |> Matrax.to_list_of_lists()
      [
          [0, 1, 2, 0, 1, 2],
          [0, 1, 2, 0, 1, 2]
      ]
  """
  @spec reshape(t, pos_integer, pos_integer) :: t
  def reshape(
        %Matrax{changes: [{:reshape, {rows, columns}} | changes_tl]} = matrax,
        desired_rows,
        desired_columns
      ) do
    reshape(
      %Matrax{matrax | rows: rows, columns: columns, changes: changes_tl},
      desired_rows,
      desired_columns
    )
  end

  def reshape(
        %Matrax{rows: rows, columns: columns, changes: changes} = matrax,
        desired_rows,
        desired_columns
      )
      when rows * columns == desired_rows * desired_columns do
    %Matrax{
      matrax
      | rows: desired_rows,
        columns: desired_columns,
        changes: [{:reshape, {rows, columns}} | changes]
    }
  end

  @doc """
  Returns position of the first occurence of the given `value`
  or `nil ` if nothing was found.

  ## Examples

      iex> Matrax.new(5, 5) |> Matrax.find(0)
      {0, 0}
      iex> matrax = Matrax.new(5, 5, seed_fun: fn _, {row, col} -> row * col end)
      iex> matrax |> Matrax.find(16)
      {4, 4}
      iex> matrax |> Matrax.find(42)
      nil
  """
  @spec find(t, integer) :: position | nil
  def find(%Matrax{min: min, max: max} = matrax, value) when is_integer(value) do
    case value do
      v when v < min or v > max ->
        nil

      _else ->
        do_find(matrax, 1, count(matrax) + 1, value)
    end
  end

  defp do_find(_, same, same, _) do
    nil
  end

  defp do_find(matrax, index, one_over_last_index, value) do
    position = index_to_position(matrax, index)

    case get(matrax, position) do
      ^value -> position
      _else -> do_find(matrax, index + 1, one_over_last_index, value)
    end
  end

  @doc """
  Flip columns of matrix in the left-right direction (vertical axis).

  After `flip_lr/1` the access path to positions will be
  modified during execution.

  If you want to get a new `:atomics` with mofified data
  use the `copy/1` function which applies the `:changes`.

  ## Examples

      iex> matrax = Matrax.new(3, 4, seed_fun: fn _, {_row, col} -> col end)
      iex> matrax |> Matrax.to_list_of_lists()
      [
          [0, 1, 2, 3],
          [0, 1, 2, 3],
          [0, 1, 2, 3]
      ]
      iex> matrax |> Matrax.flip_lr() |> Matrax.to_list_of_lists()
      [
          [3, 2, 1, 0],
          [3, 2, 1, 0],
          [3, 2, 1, 0]
      ]
  """
  @spec flip_lr(t) :: t
  def flip_lr(%Matrax{changes: [:flip_lr | changes_tl]} = matrax) do
    %Matrax{matrax | changes: changes_tl}
  end

  def flip_lr(%Matrax{changes: changes} = matrax) do
    %Matrax{matrax | changes: [:flip_lr | changes]}
  end

  @doc """
  Flip rows of matrix in the up-down direction (horizontal axis).

  After `flip_ud/1` the access path to positions will be
  modified during execution.

  If you want to get a new `:atomics` with mofified data
  use the `copy/1` function which applies the `:changes`.

  ## Examples

      iex> matrax = Matrax.new(3, 4, seed_fun: fn _, {row, _col} -> row end)
      iex> matrax |> Matrax.to_list_of_lists()
      [
          [0, 0, 0, 0],
          [1, 1, 1, 1],
          [2, 2, 2, 2]
      ]
      iex> matrax |> Matrax.flip_ud() |> Matrax.to_list_of_lists()
      [
          [2, 2, 2, 2],
          [1, 1, 1, 1],
          [0, 0, 0, 0]
      ]
  """
  @spec flip_ud(t) :: t
  def flip_ud(%Matrax{changes: [:flip_ud | changes_tl]} = matrax) do
    %Matrax{matrax | changes: changes_tl}
  end

  def flip_ud(%Matrax{changes: changes} = matrax) do
    %Matrax{matrax | changes: [:flip_ud | changes]}
  end

  @doc """
  Trace of matrix (sum of all diagonal elements).

  ## Examples

      iex> matrax = Matrax.new(5, 5, seed_fun: fn _ -> 1 end)
      iex> matrax |> Matrax.trace()
      5
  """
  @spec trace(t) :: integer()
  def trace(%Matrax{} = matrax) do
    matrax
    |> diagonal()
    |> sum()
  end

  @doc """
  Set row of a matrix at `row_index` to the values from the given 1-row matrix.

  ## Examples

      iex> matrax = Matrax.new(5, 5, seed_fun: fn _ -> 1 end)
      iex> row_matrax = Matrax.new(1, 5, seed_fun: fn _ -> 3 end)
      iex> Matrax.set_row(matrax, 2, row_matrax) |> Matrax.to_list_of_lists
      [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [3, 3, 3, 3, 3],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
      ]
  """
  @spec set_row(t, non_neg_integer, t) :: t
  def set_row(
        %Matrax{columns: columns} = matrax,
        row_index,
        %Matrax{columns: columns, rows: 1} = row_matrax
      ) do
    matrax
    |> row(row_index)
    |> Matrax.apply(fn _, position ->
      get(row_matrax, position)
    end)

    matrax
  end

  @doc """
  Set column of a matrix at `column_index` to the values from the given 1-column matrix.

  ## Examples

      iex> matrax = Matrax.new(5, 5, seed_fun: fn _ -> 1 end)
      iex> column_matrax = Matrax.new(5, 1, seed_fun: fn _ -> 3 end)
      iex> Matrax.set_column(matrax, 2, column_matrax) |> Matrax.to_list_of_lists
      [
        [1, 1, 3, 1, 1],
        [1, 1, 3, 1, 1],
        [1, 1, 3, 1, 1],
        [1, 1, 3, 1, 1],
        [1, 1, 3, 1, 1],
      ]
  """
  @spec set_column(t, non_neg_integer, t) :: t
  def set_column(
        %Matrax{rows: rows} = matrax,
        column_index,
        %Matrax{rows: rows, columns: 1} = column_matrax
      ) do
    matrax
    |> column(column_index)
    |> Matrax.apply(fn _, position ->
      get(column_matrax, position)
    end)

    matrax
  end

  @doc """
  Clears all changes made to `%Matrax{}` struct by
  setting the `:changes` key to `[]` and reverting its modifications
  to `:rows` & `:columns`.

  Clears access path only modifications like `transpose/1` but not
  modifications to integer values in the `:atomics`.

  ## Examples

      iex> matrax = Matrax.identity(3)
      iex> matrax |> Matrax.to_list_of_lists()
      [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
      ]
      iex> matrax = matrax |> Matrax.diagonal()
      iex> matrax |> Matrax.apply(fn _ -> 8 end)
      iex> matrax |> Matrax.to_list_of_lists()
      [[8, 8, 8]]
      iex> matrax = matrax |> Matrax.column(0)
      iex> matrax |> Matrax.to_list_of_lists()
      [[8]]
      iex> matrax = matrax |> Matrax.clear_changes()
      iex> matrax |> Matrax.to_list_of_lists()
      [
        [8, 0, 0],
        [0, 8, 0],
        [0, 0, 8]
      ]
  """
  @spec clear_changes(t) :: t
  def clear_changes(%Matrax{} = matrax) do
    do_clear_changes(matrax)
  end

  defp do_clear_changes(%Matrax{changes: []} = matrax) do
    matrax
  end

  defp do_clear_changes(matrax) do
    do_clear_changes(matrax |> clear_last_change())
  end

  @doc """
  Clears last change made to `%Matrax{}` struct by removing
  the head of `:changes` key and reverting its modifications
  to `:rows` & `:columns`.

  Clears access path only modifications like `transpose/1` but not
  modifications to integer values in the `:atomics`.

  ## Examples

      iex> matrax = Matrax.identity(3)
      iex> matrax |> Matrax.to_list_of_lists()
      [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
      ]
      iex> matrax = matrax |> Matrax.diagonal()
      iex> matrax |> Matrax.apply(fn _ -> 8 end)
      iex> matrax |> Matrax.to_list_of_lists()
      [[8, 8, 8]]
      iex> matrax = matrax |> Matrax.clear_last_change()
      iex> matrax |> Matrax.to_list_of_lists()
      [
        [8, 0, 0],
        [0, 8, 0],
        [0, 0, 8]
      ]
  """
  @spec clear_last_change(t) :: t
  def clear_last_change(%Matrax{changes: []} = matrax) do
    matrax
  end

  def clear_last_change(%Matrax{changes: [change | changes_tl]} = matrax) when is_atom(change) do
    %Matrax{matrax | changes: changes_tl}
  end

  def clear_last_change(%Matrax{changes: [change | changes_tl]} = matrax) when is_tuple(change) do
    {rows, columns} = elem(change, 1)

    %Matrax{matrax | rows: rows, columns: columns, changes: changes_tl}
  end

  @doc """
  Drops row of matrix at given `row_index`.

  Only modifies the struct, it doesn't move or mutate data.

  After `drop_row/2` the access path to positions
  will be modified during execution.

  If you want to get a new `:atomics` with mofified data
  use the `copy/1` function which applies the `:changes`.

  ## Examples

      iex> matrax = Matrax.new(5, 4, seed_fun: fn _, {row, _col} -> row end)
      iex> matrax |> Matrax.to_list_of_lists()
      [
          [0, 0, 0, 0],
          [1, 1, 1, 1],
          [2, 2, 2, 2],
          [3, 3, 3, 3],
          [4, 4, 4, 4],
      ]
      iex> matrax |> Matrax.drop_row(1) |> Matrax.to_list_of_lists()
      [
          [0, 0, 0, 0],
          [2, 2, 2, 2],
          [3, 3, 3, 3],
          [4, 4, 4, 4],
      ]
  """
  @spec drop_row(t, non_neg_integer) :: t
  def drop_row(%Matrax{rows: rows, changes: changes} = matrax, row_index)
      when rows > 1 and row_index >= 0 and row_index < rows do
    %Matrax{matrax | rows: rows - 1, changes: [{:drop_row, row_index} | changes]}
  end

  @doc """
  Drops column of matrix at given `column_index`.

  Only modifies the struct, it doesn't move or mutate data.

  After `drop_column/2` the access path to positions
  will be modified during execution.

  If you want to get a new `:atomics` with mofified data
  use the `copy/1` function which applies the `:changes`.

  ## Examples

      iex> matrax = Matrax.new(4, 5, seed_fun: fn _, {_row, col} -> col end)
      iex> matrax |> Matrax.to_list_of_lists()
      [
          [0, 1, 2, 3, 4],
          [0, 1, 2, 3, 4],
          [0, 1, 2, 3, 4],
          [0, 1, 2, 3, 4],
      ]
      iex> matrax |> Matrax.drop_column(1) |> Matrax.to_list_of_lists()
      [
          [0, 2, 3, 4],
          [0, 2, 3, 4],
          [0, 2, 3, 4],
          [0, 2, 3, 4],
      ]
  """
  @spec drop_column(t, non_neg_integer) :: t
  def drop_column(%Matrax{columns: columns, changes: changes} = matrax, column_index)
      when columns > 1 and column_index >= 0 and column_index < columns do
    %Matrax{matrax | columns: columns - 1, changes: [{:drop_column, column_index} | changes]}
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

    def slice(%Matrax{} = matrax) do
      {
        :ok,
        Matrax.count(matrax),
        fn start, length ->
          do_slice(matrax, start + 1, length)
        end
      }
    end

    defp do_slice(_, _, 0), do: []

    defp do_slice(matrax, index, length) do
      position = Matrax.index_to_position(matrax, index)

      [Matrax.get(matrax, position) | do_slice(matrax, index + 1, length - 1)]
    end

    def reduce(%Matrax{} = matrax, acc, fun) do
      do_reduce({matrax, 0, Matrax.count(matrax)}, acc, fun)
    end

    def do_reduce(_, {:halt, acc}, _fun), do: {:halted, acc}
    def do_reduce(tuple, {:suspend, acc}, fun), do: {:suspended, acc, &do_reduce(tuple, &1, fun)}
    def do_reduce({_, same, same}, {:cont, acc}, _fun), do: {:done, acc}

    def do_reduce({matrax, index, count}, {:cont, acc}, fun) do
      position = Matrax.index_to_position(matrax, index + 1)

      do_reduce(
        {matrax, index + 1, count},
        fun.(Matrax.get(matrax, position), acc),
        fun
      )
    end
  end
end
