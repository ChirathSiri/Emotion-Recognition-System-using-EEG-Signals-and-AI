const loginForm = document.getElementById("login-form");
const errorPopup = document.querySelector('.error-popup');
const errorContent = errorPopup.querySelector('p');
const usernameInput = document.getElementById("Username");
const passwordInput = document.getElementById("Password");

loginForm.addEventListener("submit", (event) => {
    event.preventDefault(); // Prevent default form submission

    const username = usernameInput.value.trim();
    const password = passwordInput.value.trim();

    // Send login data to server-side PHP script
    fetch('login.php', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username: username, password: password }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            window.location.href = "{{ url_for('home') }}"; // Redirect to main page on successful login
        } else {
            displayErrorPopup(data.message); // Display error message on failed login
        }
    })
    .catch(error => {
        console.error('Error:', error);
        displayErrorPopup('An error occurred. Please try again later.'); // Display generic error message
    });
});

function displayErrorPopup(errorMessage) {
    errorContent.textContent = errorMessage;
    errorPopup.style.display = 'block';
}

// Close popup on button click
errorPopup.querySelector('button').addEventListener('click', () => {
    errorPopup.style.display = 'none';
});
