defmodule MatraxTest do
  use ExUnit.Case
  doctest Matrax

  test "new returns struct" do
    assert %Matrax{atomics: atomics, rows: 50, columns: 10} = Matrax.new(50, 10, signed: false)

    assert %{min: 0, size: 500} = :atomics.info(atomics)

    assert %Matrax{atomics: atomics2, rows: 2, columns: 8} = Matrax.new(2, 8, signed: true)

    assert %{min: -9_223_372_036_854_775_808, size: 16} = :atomics.info(atomics2)
  end

  test "seeds & gets values" do
    matrax = Matrax.new(10, 5, seed_fun: fn _, {row, col} -> row * col end)

    1..50
    |> Enum.each(fn index ->
      {row, col} = Matrax.index_to_position(matrax, index)

      assert row * col == Matrax.get(matrax, {row, col})
    end)
  end

  test "puts values" do
    matrax = Matrax.new(10, 10)

    20..40
    |> Enum.each(fn index ->
      position = Matrax.index_to_position(matrax, index)

      Matrax.put(matrax, position, index)

      assert index == Matrax.get(matrax, position)
    end)
  end

  test "column & reshape interop" do
    matrax = Matrax.new(5, 5, seed_fun: fn _, {row, _col} -> row end)

    col2 = matrax |> Matrax.column(2)

    assert [[0], [1], [2], [3], [4]] == col2 |> Matrax.to_list_of_lists()

    assert [[0, 1, 2, 3, 4]] == col2 |> Matrax.reshape(1, 5) |> Matrax.to_list_of_lists()
  end
end
