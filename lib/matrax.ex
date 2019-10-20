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

  @enforce_keys [:atomics, :rows, :columns, :min, :max]
  defstruct [:atomics, :rows, :columns, :min, :max]

  @type t :: %__MODULE__{
          atomics: reference,
          rows: pos_integer,
          columns: pos_integer,
          min: integer,
          max: pos_integer
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

    matrax = %Matrax{atomics: atomics, rows: rows, columns: columns, min: min, max: max}

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
  def index_to_position(%Matrax{columns: columns}, index) when is_integer(index) do
    index_to_position(columns, index)
  end

  def index_to_position(columns, index) do
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
  def position_to_index(%Matrax{columns: columns}, {row, col}) do
    row * columns + col + 1
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
  Adds `incr` to atomic at `position` and return result.

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
  Returns size (rows * columns) of matrax.

  ## Examples

      iex> matrax = Matrax.new(5, 5)
      iex> Matrax.size(matrax)
      25
      iex> Matrax.size(matrax) == :atomics.info(matrax.atomics).size
      true
  """
  @spec size(t) :: pos_integer
  def size(%Matrax{rows: rows, columns: columns}) do
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
    last_index = size(matrax)

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
    last_index = size(matrax)

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
    last_index = size(matrax)

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

    do_apply(matrax.atomics, matrax.columns, size(matrax), fun_arity, fun)
  end

  defp do_apply(_, _, 0, _, _), do: :ok

  defp do_apply(atomics, columns, index, fun_arity, fun) do
    value =
      case fun_arity do
        1 -> fun.(:atomics.get(atomics, index))
        2 -> fun.(:atomics.get(atomics, index), index_to_position(columns, index))
      end

    :atomics.put(atomics, index, value)

    do_apply(atomics, columns, index - 1, fun_arity, fun)
  end

  @doc """
  Converts `%Matrax{}` struct to list of lists.

  ## Examples

      iex> m = Matrax.new(5, 5, seed_fun: fn _, {row, col} -> row * col end)
      iex> Matrax.to_list_of_lists(m)
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
  Checks if `integer` exists within `matrax`.

      iex> matrax = Matrax.new(5, 5, seed_fun: fn _, {row, col} -> row * col end)
      iex> matrax |> Matrax.member?(6)
      true
      iex> matrax |> Matrax.member?(100)
      false
  """
  @spec member?(t, integer) :: boolean
  def member?(%Matrax{min: min, max: max} = matrax, integer) when is_integer(integer) do
    case integer do
      i when i < min or i > max ->
        false

      _else ->
        do_member?(matrax.atomics, size(matrax), integer)
    end
  end

  defp do_member?(_, 0, _), do: false

  defp do_member?(atomics, index, integer) do
    case :atomics.get(atomics, index) do
      ^integer -> true
      _else -> do_member?(atomics, index - 1, integer)
    end
  end

  defimpl Enumerable do
    @moduledoc false

    alias Matrax

    def count(%Matrax{} = matrax) do
      {:ok, Matrax.size(matrax)}
    end

    def member?(%Matrax{} = matrax, int) do
      {:ok, Matrax.member?(matrax, int)}
    end

    def slice(%Matrax{atomics: atomics} = matrax) do
      {
        :ok,
        Matrax.size(matrax),
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
      size = Matrax.size(matrax)

      do_reduce({atomics, 0, size}, acc, fun)
    end

    def do_reduce(_, {:halt, acc}, _fun), do: {:halted, acc}
    def do_reduce(tuple, {:suspend, acc}, fun), do: {:suspended, acc, &do_reduce(tuple, &1, fun)}
    def do_reduce({_, size, size}, {:cont, acc}, _fun), do: {:done, acc}

    def do_reduce({atomics, index, size}, {:cont, acc}, fun) do
      do_reduce(
        {atomics, index + 1, size},
        fun.(:atomics.get(atomics, index + 1), acc),
        fun
      )
    end
  end
end
