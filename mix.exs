defmodule Matrax.MixProject do
  use Mix.Project

  @version "0.3.4"
  @github "https://github.com/preciz/matrax"

  def project do
    [
      app: :matrax,
      version: @version,
      elixir: "~> 1.7",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      docs: docs(),
      package: package(),
      homepage_url: @github,
      description: """
      A matrix library in pure Elixir based on atomics.
      """
    ]
  end

  def application do
    []
  end

  defp deps do
    [
      {:ex_doc, "~> 0.21", only: :dev, runtime: false},
    ]
  end

  defp docs do
    [
      main: "Matrax",
      source_ref: "v#{@version}",
      source_url: @github,
    ]
  end

  defp package do
    [
      maintainers: ["Barna Kovacs"],
      licenses: ["MIT"],
      links: %{"GitHub" => @github}
    ]
  end
end
