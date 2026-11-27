# Forward values for grad's n-dimensional ops, from base R in double precision.
#
# Reads fixtures/nd-cases.json (written by nd_reference.py) and writes
# fixtures/nd-r.json: the output of every case, computed with R's own array
# machinery — aperm, apply, %*%, and array indexing — so that layout and axis
# conventions are checked against a column-major system as well as against
# PyTorch. Arrays in the fixtures are row-major; toR/fromR convert.
#
# Run from the package root:  Rscript tests/references/nd_reference.R

library(jsonlite)

fx <- fromJSON("tests/fixtures/nd-cases.json", simplifyVector = FALSE)

# Row-major flat data + shape -> an R array whose x[i, j, ...] is that element.
toR <- function(t) {
  s <- as.integer(unlist(t$shape))
  d <- as.numeric(unlist(t$data))
  if (length(s) <= 1) return(array(d, dim = max(1L, length(d))))
  aperm(array(d, dim = rev(s)), rev(seq_along(s)))
}
# An R array (or a bare number, for a whole reduction) -> row-major data + shape.
fromR <- function(x, scalar = FALSE) {
  if (scalar) return(list(shape = integer(0), data = as.numeric(x)))
  s <- dim(x)
  if (is.null(s)) s <- length(x)
  list(shape = as.integer(s), data = as.numeric(aperm(array(x, dim = s), rev(seq_along(s)))))
}
dims <- function(x) if (is.null(dim(x))) length(x) else dim(x)

# numpy's broadcast shape, aligned from the right.
bshape <- function(da, db) {
  r <- max(length(da), length(db))
  da <- c(rep(1L, r - length(da)), da)
  db <- c(rep(1L, r - length(db)), db)
  ifelse(da == 1L, db, da)
}
# Repeat x along its length-1 axes to `out` dims, by R's matrix indexing.
expand <- function(x, out) {
  xd <- dims(x)
  xd <- c(rep(1L, length(out) - length(xd)), xd)
  x <- array(x, dim = xd) # leading 1s keep every column-major offset
  idx <- as.matrix(expand.grid(lapply(out, seq_len)))
  for (k in seq_along(out)) if (xd[k] == 1L) idx[, k] <- 1L
  array(x[idx], dim = out)
}
index_all <- function(x, idx) do.call(`[`, c(list(x), idx, list(drop = FALSE)))

run <- function(c) {
  a <- toR(c$inputs$a)
  args <- c$args
  op <- c$op
  if (op %in% c("add", "sub", "mul", "div", "maximum")) {
    b <- toR(c$inputs$b)
    out <- bshape(dims(a), dims(b))
    A <- expand(a, out)
    B <- expand(b, out)
    y <- switch(op, add = A + B, sub = A - B, mul = A * B, div = A / B,
                maximum = array(pmax(A, B), dim = out))
    return(fromR(y))
  }
  if (op %in% c("sum", "mean", "max")) {
    f <- switch(op, sum = sum, mean = mean, max = max)
    if (is.null(args$axis)) return(fromR(f(a), scalar = TRUE))
    d <- dims(a)
    k <- args$axis + 1L
    keep <- setdiff(seq_along(d), k)
    y <- apply(a, keep, f)
    nd <- if (isTRUE(args$keepDims)) replace(d, k, 1L) else d[keep]
    return(fromR(array(y, dim = nd)))
  }
  if (op == "transpose") {
    perm <- if (is.null(args$perm)) rev(seq_along(dims(a))) else as.integer(unlist(args$perm)) + 1L
    return(fromR(aperm(a, perm)))
  }
  if (op == "matmul") {
    b <- toR(c$inputs$b)
    da <- dims(a)
    db <- dims(b)
    r <- length(da)
    lead <- da[seq_len(r - 2)]
    m <- da[r - 1]
    k <- da[r]
    vec <- length(db) == 1
    n <- if (vec) 1L else db[length(db)]
    batched <- length(db) >= 3
    outd <- if (vec) c(lead, m) else c(lead, m, n)
    y <- array(0, dim = outd)
    grid <- if (length(lead)) as.matrix(expand.grid(lapply(lead, seq_len))) else matrix(0L, 1, 0)
    for (g in seq_len(nrow(grid))) {
      li <- as.list(grid[g, ])
      A <- matrix(as.vector(index_all(a, c(li, list(TRUE, TRUE)))), nrow = m)
      B <- if (vec) matrix(as.vector(b), ncol = 1) else if (batched)
        matrix(as.vector(index_all(b, c(li, list(TRUE, TRUE)))), nrow = k) else b
      P <- A %*% B
      y <- do.call(`[<-`, c(list(y), li, if (vec) list(TRUE) else list(TRUE, TRUE), list(value = if (vec) as.vector(P) else P)))
    }
    return(fromR(y))
  }
  if (op == "slice") {
    st <- as.integer(unlist(args$start))
    sz <- as.integer(unlist(args$size))
    return(fromR(index_all(a, lapply(seq_along(st), function(d) (st[d] + 1L):(st[d] + sz[d])))))
  }
  if (op == "concat") {
    b <- toR(c$inputs$b)
    k <- args$axis + 1L
    da <- dims(a)
    db <- dims(b)
    out <- replace(da, k, da[k] + db[k])
    y <- array(0, dim = out)
    part <- function(y, x, from) {
      idx <- lapply(out, function(n) TRUE)
      idx[[k]] <- from + seq_len(dims(x)[k])
      do.call(`[<-`, c(list(y), idx, list(value = x)))
    }
    return(fromR(part(part(y, a, 0L), b, da[k])))
  }
  if (op == "reshape") {
    return(list(shape = as.integer(unlist(args$shape)), data = fromR(a)$data))
  }
  if (op == "gather") {
    ix <- fromR(toR(c$inputs$index))
    rows <- as.integer(ix$data) + 1L
    rest <- dims(a)[-1]
    picked <- if (length(rest)) index_all(a, c(list(rows), lapply(rest, function(n) TRUE))) else a[rows]
    return(list(shape = c(ix$shape, rest), data = fromR(picked)$data))
  }
  stop(paste("unknown op", op))
}

results <- lapply(fx$cases, function(c) c(list(name = c$name), list(output = run(c))))
meta <- list(generator = "nd_reference.R", R = R.version.string,
             jsonlite = as.character(packageVersion("jsonlite")), dtype = "double", order = "row-major")
writeLines(toJSON(list(meta = meta, results = results), auto_unbox = TRUE, digits = NA, always_decimal = TRUE),
           "tests/fixtures/nd-r.json")
cat(length(results), "cases written\n")
