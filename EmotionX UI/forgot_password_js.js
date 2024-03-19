const emailInput = document.getElementById("email");
const submitButton = document.getElementById("submit-button");
const form = document.getElementById("forgot-password-form");

emailInput.addEventListener("input", () => {
  const emailValue = emailInput.value.trim();
  if (emailValue.length > 0 && validateEmail(emailValue)) {
    submitButton.disabled = false;
  } else {
    submitButton.disabled = true;
  }
});

// Simulate form submission for demonstration
form.addEventListener("submit", (event) => {
  event.preventDefault(); // Prevent actual form submission

  // Simulate sending data to server
  const email = emailInput.value.trim();
  console.log(`Sending password reset request for email: ${email}`);

  // Simulate successful response
  alert("Password reset email sent! Check your inbox for instructions.");
  emailInput.value = ""; // Clear input field
  submitButton.disabled = true; // Disable button again
});

function validateEmail(email) {
  // Add your email validation logic here
  // For example, a simple regular expression check:
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}