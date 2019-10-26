# Changelog for Matrax

## v0.3.0
  * BREAKING - `argmax/1` now returns first position instead of last given multiple occurence of equal maximal values
  * Implemented `argmin/1` function
  * Implemented `clear_changes/1` function (for reverting all changes in :changes key)
  * Implemented `clear_last_change/1` function (for reverting last change in :changes key)

## v0.2.4
  * Implemented `trace/1` function
  * Implemented `set_column/3` function
  * Implemented `set_row/3` function

## v0.2.3
  * Implemented `new/1` with list_of_lists as argument
  * Improved overall performance and more than halved copy/1 runtime
  * Implemented `row/2` function
  * Implemented `column/2` function

## v0.2.2
  * Implemented `flip_ud/1` function

## v0.2.1
  * Implemented `flip_lr/1` function

## v0.2.0
  * BREAKING - changed `submatrix/3` function to only modify access path (can be finished with `copy/1`)
  * Implemented `reshape/3` function
  * Implemented `identity/1` function
  * Implemented `argmax/1` function
  * Implemented `find/2` function
  * Implemented `diagonal/1` function
