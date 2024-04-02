<?php
// Establish connection to the database
$servername = "localhost";
$username = "root"; // Change this to your MySQL username
$password = ""; // Change this to your MySQL password
$dbname = "USERS"; // Change this to your MySQL database name

$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Retrieve username and password from POST request
$username = $_POST['username'];
$password = $_POST['password'];

// Query the database to check if the provided credentials are valid
$sql = "SELECT * FROM user_details WHERE username='$username' AND password='$password'";
$result = $conn->query($sql);

if ($result->num_rows > 0) {
    // User exists, login successful
    echo json_encode(array("success" => true));
} else {
    // User does not exist or invalid credentials
    echo json_encode(array("success" => false));
}

$conn->close();
?>