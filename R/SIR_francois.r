
setwd("C:/Users/franc/Documents/GitHub/PI/R")
library(readr)
dataframe <- read_csv("https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv")

SIR <- function(time, state, parameters) {
  par <- as.list(c(state, parameters))
  with(par, {
    dS <- -beta * I * S / (S + I + R)
    dI <- beta * I * S / (S + I + R) - gamma * I
    dR <- gamma * I
    list(c(dS, dI, dR))
  })
}

# Transforme en matrice:
Infected <- dataframe$num_positive
Infected[1] <- 1
set_size <- dim(data)[1]
# Make cumulative:
#for(i in (2: length(Infected))){
#  Infected[i] <- Infected[i-1] + Infected[i]
#}

# Time vector
Day <- 1:(length(Infected))

# Initial state:
N <- 1000000
init <- c(
  S = N - Infected[1],
  I = Infected[1],
  R = Infected[1]
)

# Needed for ODE
library(deSolve)
# Least square error to optimie
RSS <- function(parameters) {
  names(parameters) <- c("beta", "gamma")
  out <- ode(y = init, times = Day, func = SIR, parms = parameters)

  fit <- out[, 3]
  print(fit)
  print(" - ")
  print(Infected)
  print("\n")
  sum((Infected - out[, 2] - out[, 3])^2)
}
test_time <- 1:100
params <- c(0.5, 0.5)
names(params) <- c("beta", "gamma")
test_predictions <-ode(y = init, times = test_time, func = SIR, parms = params)
for(i in (0: len(test_time))){

}
plot(test_predictions[, 1])
plot(test_predictions[, 2])
plot(test_predictions[, 3])

# Optimization:
Opt <- optim(c(0.5, 0.5),
             RSS,
             method = "L-BFGS-B",
             lower = c(0, 0),
             upper = c(1, 1)
)

# check for convergence
Opt$message





