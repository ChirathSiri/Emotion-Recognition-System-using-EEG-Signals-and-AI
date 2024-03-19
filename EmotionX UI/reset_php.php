<?php
// Start session to store temporary data
session_start();

// Check if the form was submitted
if (isset($_POST['email'])) {
  $email = $_POST['email'];

  // Validate email address (add your validation logic here)
  if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
    // Invalid email address
    $_SESSION['error'] = "Please enter a valid email address.";
    header('Location: Forgot_Password.html');
    exit();
  }

  // Connect to your database

  
$db = new mysqli('localhost', 'username', 'password', 'database_name');
  if ($db->connect_error) {
    die("Connection failed: " . $db->connect_error);
  }

  // Check if the email exists in your database
  $sql = "SELECT * FROM users WHERE email = '$email'";
  $result = $db->query($sql);

  if ($result->num_rows == 1) {
    // Email found, generate a unique password reset token
    $token = bin2hex(random_bytes(32));  // Generate a secure token

    // Store the token in the database, associated with the user's email
    $sql = "UPDATE users SET reset_token='$token', reset_expires=DATE_ADD(NOW(), INTERVAL 1 HOUR) WHERE email='$email'";
    $db->query($sql);

    // Send an email with a link to reset the password
    $link = "file:///Users/chirathsirimanna/Downloads/Final-Year-Project/Final-Year-Project/Login.html#/reset_php.php?token=$token";
    $message = "Click this link to reset your password: $link";
    mail($email, "Password Reset Request", $message);

    $_SESSION['success'] = "Password reset email sent! Check your inbox for instructions.";
    header('Location: Forgot_Password.html');
  } else {
    // Email not found
    $_SESSION['error'] = "Email address not found.";
    header('Location: Forgot_Password.html');
  }

  $db->close();
} else {
  // Form not submitted, redirect back to forgot password page
  header('Location: Forgot_Password.html');
}
?>