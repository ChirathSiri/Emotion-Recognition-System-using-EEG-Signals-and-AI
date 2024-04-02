<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $targetDir = "uploads/";
    $targetFile = $targetDir . basename($_FILES["bdfFile"]["name"]);
    $uploadOk = 1;
    $fileType = strtolower(pathinfo($targetFile,PATHINFO_EXTENSION));

    // Check if file is a valid BDF file
    if($fileType != "bdf") {
        echo "Sorry, only BDF files are allowed.";
        $uploadOk = 0;
    }

    // Check if file already exists
    if (file_exists($targetFile)) {
        echo "Sorry, file already exists.";
        $uploadOk = 0;
    }

    // Move the uploaded file
    if ($uploadOk == 0) {
        echo "Sorry, your file was not uploaded.";
    } else {
        if (move_uploaded_file($_FILES["bdfFile"]["tmp_name"], $targetFile)) {
            echo "The file ". htmlspecialchars( basename( $_FILES["bdfFile"]["name"])). " has been uploaded.";
            // Redirect to the result page
            echo "<script>window.location.href = '_result_page_.html';</script>";
            exit(); // Stop further execution
        } else {
            echo "Sorry, there was an error uploading your file.";
        }
    }
}
?>