<?php
// Connect to your database
$servername = "localhost";
$username = "root"; // Your MySQL username
$password = ""; // Your MySQL password
$dbname = "USERS"; // Your MySQL database name

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
  die("Connection failed: " . $conn->connect_error);
}

// Check if form is submitted
if ($_SERVER["REQUEST_METHOD"] == "POST") {
  // Prepare and bind parameters using a prepared statement to prevent SQL injection
  $stmt = $conn->prepare("INSERT INTO user_details (user_id, name, email, username, password) VALUES (?, ?, ?, ?, ?)");
  $stmt->bind_param("sssss", $user_id, $name, $email, $username, $password);

  // Escape user inputs for security
  $user_id = $_POST['user_id'];
  $name = $_POST['name'];
  $email = $_POST['email'];
  $username = $_POST['username'];
  $password = $_POST['password'];

  // Attempt to insert data
  if ($stmt->execute()) {
    // Redirect to the page with success message
    header("Location: create_account.php?success=true");
    exit();
  } else {
    // Redirect to the page with error message
    header("Location: create_account.php?error=true");
    exit();
  }

  // Close statement
  $stmt->close();
}

// Close connection
$conn->close();
?>