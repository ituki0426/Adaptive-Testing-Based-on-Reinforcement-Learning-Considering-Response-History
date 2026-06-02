Response <- function(a, b, c, theta, D=1) {
  p<-(1-c)/(1+exp(-D*a*(theta-b)))+c
  as.numeric(runif(1)<=p) }

MFI <- function(a, b, c, theta, D=1) { 
  D^2*a^2*(1-c) / (c+exp(D*a*(theta-b))) / (1+exp(-D*a*(theta-b)))^2 }

KLP <- function(ans_un, theta_est, ans_ed, resp, D=1, weighted=F) {
  
  p <- function(theta, a, b, c, D){ (1-c)/(1+exp(-D*a*(theta-b)))+c }  
  
  notes<-seq(-3, 3, 0.25)
  
  a <- ans_un[,1];  b <- ans_un[,2];  c <- ans_un[,3]
  kl_note <- function(note) { p(theta_est, a, b, c, D) * log(p(theta_est, a, b, c, D) / p(note, a, b, c, D)) + (1 - p(theta_est, a, b, c, D)) * log((1 - p(theta_est, a, b, c, D))/(1 - p(note, a, b, c, D))) }
  kl_note_value <-apply(matrix(notes), 1, kl_note)
  
  a <- ans_ed[,1];  b <- ans_ed[,2];  c <- ans_ed[,3]  
  Likelihood <- function(theta) { prod(p(theta, a, b, c, D)^resp * (1-p(theta, a, b, c, D))^(1-resp)) }
  lik <- apply(matrix(notes), 1, Likelihood)
  
  if (weighted==T) weight <- dnorm(notes) else weight <- 1
  
  integ <- function(y)  { y %*% (lik * weight) *(notes[2]-notes[1]) }
  info <- apply(matrix(kl_note_value, nrow=nrow(ans_un)), 1, integ)
  return(info)
}

MLWI <- function(item_un, items_ed=NULL, resp=NULL, D=1, MPWI=F) {
  Fisher <- function(x) {
    a <- item_un[,1];    b <- item_un[,2];    c <- item_un[,3]
    D^2*a^2*(1-c) / (c+exp(D*a*(x-b))) / (1+exp(-D*a*(x-b)))^2
  }
  
  X <- seq(-3, 3, 0.25)
  Lik <- function(x) {
    a <- items_ed[,1]; b <- items_ed[,2]; c <- items_ed[,3]
    p <- (1-c)/(1+exp(-D*a*(x-b)))+c
    
    res <- prod(p^resp*(1-p)^(1-resp))
    if (MPWI == T) res <- res * (2*pi)^-0.5 * exp(-x^2/2)
    
    return(res)
  }
  lx <- sapply(X, Lik)
  
  integrate <-function(ix){
    fx <- lx * ix
    sum((fx[-1]+fx[-25])/2*0.25)
  }
  
  info <- apply(matrix(sapply(X, Fisher), nrow=nrow(item_un)), 1, integrate)
  
  return(info)
}

MEI <- function (itemBank, it.given, resp, theta, D=1) { 
  
  if (nrow(it.given)<1) { it.given=NULL
  } else { it.given <- cbind(it.given, d=1) }
  
  itemBank <- cbind(itemBank, d=1)
  
  th0 <- apply(itemBank, 1, function(para) MAP(rbind(it.given, para), c(resp, 0), D=D) )
  th1 <- apply(itemBank, 1, function(para) MAP(rbind(it.given, para), c(resp, 1), D=D) )
  
  a = itemBank[,1]; b = itemBank[,2]; c = itemBank[,3]
  
  p1 <- (1-c)/(1+exp(-D*a*(theta-b)))+c
  p0 <- 1 - p1
  
  Ij0 <- D^2*a^2*(1-c) / (c+exp(D*a*(th0-b))) / (1+exp(-D*a*(th0-b)))^2
  Ij1 <- D^2*a^2*(1-c) / (c+exp(D*a*(th1-b))) / (1+exp(-D*a*(th1-b)))^2
  
  res <- p0 * Ij0 + p1 * Ij1
  
  return(as.numeric(res))
}

MLE<-function(item_paras, resp, D=1){
  a <- item_paras[,1]
  b <- item_paras[,2]
  c <- item_paras[,3]
  
  likelihood <- function(x){
    p <- (1-c)/(1+exp(-D*a*(x-b)))+c
    prod(p^resp*(1-p)^(1-resp))
  }
  
  optimize(likelihood, c(-4,4), maximum = TRUE)$maximum
}

EAP<-function(item_paras, resp, D=1){
  a <- item_paras[,1]
  b <- item_paras[,2]
  c <- item_paras[,3]
  
  theta_x<-seq(-4, 4, length.out=33)
  
  gx<-dnorm(theta_x) 

  wx<-gx/sum(gx)
  
  likelihood <- function(x){
    p<-(1-c)/(1+exp(-D*a*(x-b)))+c
    prod(p^resp*(1-p)^(1-resp))
  }
  Lx<-apply(matrix(theta_x), 1, likelihood)
  
  sum(theta_x*Lx*wx)/sum(Lx*wx)
}

MAP <- function(it.given, x, range=c(-4,4), D = 1) {
  
  a <- it.given[,1]
  b <- it.given[,2]
  c <- it.given[,3]
  
  likelihood <- function(th){
    p <- (1-c)/(1+exp(-D*a*(th-b)))+c
    prod(p^x*(1-p)^(1-x)) * (2*pi)^-0.5 * exp(-th^2/2)
  }
  
  optimize(likelihood, range, maximum = TRUE)$maximum
  
}

