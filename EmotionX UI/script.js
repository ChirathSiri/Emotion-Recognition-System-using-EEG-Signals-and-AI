const loginForm = document.getElementById("login-form");
const usernameInput = document.getElementById("Username");
const passwordInput = document.getElementById("Password");

const usernameError = document.getElementById("username-error");
const passwordError = document.getElementById("password-error");
const mainContent = document.getElementById("main-content");

function displayErrorPopup(errorMessage) {
  const errorPopup = document.querySelector('.error-popup');
  const errorContent = errorPopup.querySelector('p');
  errorContent.textContent = errorMessage;
  errorPopup.classList.add('show');

  // Close popup on button click
  errorPopup.querySelector('button').addEventListener('click', () => {
    errorPopup.classList.remove('show');
  });
}

loginForm.addEventListener("submit", (event) => {
  event.preventDefault(); // Prevent default form submission

  const username = usernameInput.value.trim();
  const password = passwordInput.value.trim();
  
  let UserName = "name";
  let PassWord = "password";

  // Basic validation (replace with more robust validation later)
  //let isValid = true;
  
  if (username !== UserName || password !== PassWord){
      // Display a general error message if both username and password are invalid
      displayErrorPopup("Invalid username or password. Please try again.");
  }else {
      // Simulate successful login (replace with actual authentication logic)
      loginForm.style.display = "none";
      mainContent.style.display = "block";
      window.location.href = "main-page.html"; 
  }
  
  if (username !== UserName) {
    displayErrorPopup("Invalid username"); // Use this for popup instead
    //usernameError.textContent = "Invalid username";
	
  } else if (password !== PassWord) {
    displayErrorPopup("Invalid password"); // Use this for popup instead
    //passwordError.textContent = "Invalid password";
	
  } else  {
    // Simulate successful login (replace with actual authentication logic)
    loginForm.style.display = "none";
    mainContent.style.display = "block";
    window.location.href = "main-page.html";
  } 
});