# Load external R functions from a file
source("Functions.R")

# Load item parameters, true theta values, and responses from files
item_bank <- read.table(" ")
theta_true <- read.table(" ")
responses <- read.table(" ")

# Set parameters for the simulation
test_length <- 40  # Number of items in CAT
sample_size <- 5000  # Number of examinees

# Specify the item selection method to be used
method <- 'MFI'  # Options include: 'MFI', 'KLP', 'MPWI', 'MLWI', 'MEI'

# Prepare a dataframe to store results
cat_data <- data.frame()

# Simulate the adaptive test for each examinee
for (j in 1:sample_size) {
  
  # Initialize variables for the current examinee
  userID <- j
  bank <- item_bank  # Available items
  itemID <- resp <- theta_est <- item_id <- c()  # Responses, estimated abilities, and selected items
  
  # Administer each item
  for (i in 1:test_length) {
    
    # Initialize theta for the first item
    if (i==1) theta_est[1]<-runif(1, -0.5, 0.5)  # Random start between -0.5 and 0.5
    
    # Select the next item based on the specified method
    if (method=="MFI") {
      item_id <- which.max(MFI(bank[,1], bank[,2], bank[,3], theta_est[i]))
    } else if (method=='KLP') {
      item_id <- which.max(KLP(bank, theta_est[i], item_bank[itemID,], resp, weighted=T))
    } else if (method=='MLWI') {
      item_id <- which.max(MLWI(bank, item_bank[itemID,], resp))
    } else if (method=='MPWI') {
      item_id <- which.max(MLWI(bank, item_bank[itemID,], resp, MPWI=T))
    } else if (method=='MEI') {
      item_id <- which.max(MEI(bank, item_bank[itemID,], resp, theta_est[i]))
    }
    
    # Record the ID of the selected item
    itemID[i] <- which(item_bank[,2] == bank[item_id,2])
    
    # Record the examinee's response, either from real data or simulated based on the true theta
    if (real_resp) {
      resp[i] <- responses[j, itemID[i]]  # Real response data
    } else {
      resp[i] <- Response(bank[item_id,1], bank[item_id,2], bank[item_id,3], theta_true[j])  # Simulated response
    }
    
    # Update the ability estimate based on the response pattern
    if (all(resp == 1)) { 
      # If all responses are correct, adjust theta upwards
      theta_est[i+1] <- theta_est[i] + (max(bank[,2]) - theta_est[i]) / 2
    } else if (all(resp == 0)) { 
      # If all responses are incorrect, adjust theta downwards
      theta_est[i+1] <- theta_est[i] + (min(bank[,2]) - theta_est[i]) / 2
    } else {
      # Otherwise, use Maximum Likelihood Estimation (MLE) for updating theta
      theta_est[i+1] <- MLE(item_bank[itemID,], resp)
    }
    
    # Remove the selected item from the item bank
    bank <- bank[-item_id,]
  }
  
  # Compile results for the current examinee
  cat_data <- rbind(cat_data, data.frame(userID, step=1:test_length, itemID, resp, theta_est=theta_est[-1], bias=theta_est[-1]-theta_true[j]))
}

# Save the results to a CSV file
write.csv(cat_data, paste0("cat_data_", method, ".csv"))
