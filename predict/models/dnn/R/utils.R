cell_type <- function(x) {
  x <- as.vector(x)
  x <- factor(grepl('2i', x), levels=c(T, F), labels=c('2i', 'serum'))
  return (x)
}
