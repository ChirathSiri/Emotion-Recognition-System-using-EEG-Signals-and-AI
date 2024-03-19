<?php
// Start session to store temporary data
session_start();

// Check if the form was submitted
if (isset($_POST['username'], $_POST['email'], $_POST['password'], $_POST['confirm-password'])) {
  // Retrieve form data
  $username = $_POST['username'];
  $email = $_POST['email'];
  $password = $_POST['password'];
  $confirmPassword = $_POST['confirm-password'];

  // Validate user input (add your validation logic here)
  if (empty($username) || empty($email) || empty($password) || empty($confirmPassword)) {
    $_SESSION['error'] = "Please fill in all required fields.";
    header('Location: create-account.html');
    exit();
  }

  // Check for unique username and valid email format (replace with your validation logic)
  // ...

  // Check if passwords match
  if ($password !== $confirmPassword) {
    $_SESSION['error'] = "Passwords do not match.";
    header('Location: create-account.html');
    exit();
  }

  // Connect to your database

  
$db = new mysqli('localhost', 'username', 'password', 'database_name');
  if ($db->connect_error) {
    die("Connection failed: " . $db->connect_error);
  }

  // Hash the password securely
  $hashedPassword = password_hash($password, PASSWORD_DEFAULT);

  // Prepare SQL statement to insert user data
  $sql = "INSERT INTO users (username, email, password) VALUES (?, ?, ?)";
  $stmt = $db->prepare($sql);
  $stmt->bind_param("sss", $username, $email, $hashedPassword);

  // Execute the SQL statement
  if ($stmt->execute()) {
    $_SESSION['success'] = "Account created successfully!";
    header('Location: login.php'); // Redirect to login page (example)
  } else {
    $_SESSION['error'] = "Failed to create account. Please try again.";
    header('Location: create-account.html');
  }

  $stmt->close();
  $db->close();
} else {
  // Form not submitted, redirect back to create account page
  header('Location: create-account.html');
}
?>