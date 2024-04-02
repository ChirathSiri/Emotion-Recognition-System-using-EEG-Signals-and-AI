from flask import Flask, render_template, request, redirect, url_for
from EmotionXSystem import read_preprocessed_data, feature_extraction, create_labels, \
    map_emotion_label, predict_with_RF, load_RF_model, analyze_health_implications
from flask import Flask, render_template, request, redirect, url_for
import pymysql

app = Flask(__name__)

# Configure MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'  # Change this to your MySQL username
app.config['MYSQL_PASSWORD'] = ''  # Change this to your MySQL password
app.config['MYSQL_DB'] = 'USERS'   # Change this to your MySQL database name

# Create a MySQL connection
mysql = pymysql.connect(
    host=app.config['MYSQL_HOST'],
    user=app.config['MYSQL_USER'],
    password=app.config['MYSQL_PASSWORD'],
    db=app.config['MYSQL_DB']
)

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/home')
def home():
    return render_template('main-page.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if username and password match in the database
        cur = mysql.cursor()
        cur.execute("SELECT * FROM user_details WHERE username=%s AND password=%s", (username, password))
        user = cur.fetchone()
        cur.close()

        if user:
            # Authentication successful
            return redirect(url_for('home'))
        else:
            # Authentication failed
            error = 'Invalid username or password. Please try again.'
            return render_template('Login.html', error=error)

    return render_template('Login.html')

@app.route('/forgot', methods=['GET', 'POST'])
def forgot():
    if request.method == 'POST':
        username = request.form['username']
        new_password = request.form['new_password']

        # Check if the username exists in the database
        cur = mysql.cursor()
        cur.execute("SELECT * FROM user_details WHERE username=%s", (username,))
        existing_user = cur.fetchone()
        cur.close()

        if not existing_user:
            error = 'Username does not exist. Please enter a valid username.'
            return render_template('forgot_password.html', error=error)

        # Check if the new password is the same as the old one
        cur = mysql.cursor()
        cur.execute("SELECT password FROM user_details WHERE username=%s", (username,))
        old_password = cur.fetchone()[0]
        cur.close()

        if old_password == new_password:
            error = 'New password cannot be the same as the old one. Please choose a different password.'
            return render_template('forgot_password.html', error=error)

        # Update user's password in the database
        cur = mysql.cursor()
        cur.execute("UPDATE user_details SET password=%s WHERE username=%s", (new_password, username))
        mysql.commit()
        cur.close()

        # Render the forgot_password.html template with a success message
        message = 'Password successfully changed. You can now login with your new password.'
        return render_template('forgot_password.html', message=message)

    return render_template('forgot_password.html')

@app.route('/create', methods=['GET', 'POST'])
def create():
    if request.method == 'POST':
        user_id = request.form['user_id']
        name = request.form['name']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']

        # Check if username already exists in the database
        cur = mysql.cursor()
        cur.execute("SELECT * FROM user_details WHERE username = %s", (username,))
        existing_user = cur.fetchone()
        cur.close()

        if existing_user:
            # Username already exists, show error
            return render_template('create_account.html', error='Username already exists. Please choose a different one.')

        # Insert new user into the database
        cur = mysql.cursor()
        cur.execute("INSERT INTO user_details (user_id, name, email_address, username, password) VALUES (%s, %s, %s, %s, %s)",
                    (user_id, name, email, username, password))
        mysql.commit()
        cur.close()

        message = 'Account Successfully Created !. You can now login with your new account !.'
        return render_template('create_account.html', message=message)  # Redirect to login page after successful account creation

    return render_template('create_account.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Save the file
            uploaded_file.save(uploaded_file.filename)
            # Preprocess the data
            labels_for_feature, data_for_feature = read_preprocessed_data(uploaded_file.filename)

            if labels_for_feature is not None and data_for_feature is not None:
                eeg_band_data = feature_extraction(labels_for_feature, data_for_feature)
                labels_for_classification = create_labels(labels_for_feature)
                # classification with random_forest classification

                # Load saved models
                RF_model = load_RF_model('Model.pkl')

                # Make predictions
                RF_predictions = predict_with_RF(RF_model, eeg_band_data)

                # Map numerical labels to emotion types
                predicted_emotion_types = [map_emotion_label(label) for label in RF_predictions]

                # Return the predicted emotion type
                common_emotion = max(set(predicted_emotion_types), key=predicted_emotion_types.count)
                print("Predicted emotion type:", common_emotion)

                # Analyze health implications of predicted emotions
                suggestion = analyze_health_implications(common_emotion)

                return render_template('_result_page_.html', emotion=common_emotion, suggestion=suggestion)
            else:
                return "Error in preprocessing the data."
        else:
            return "No file uploaded."
    else:
        return render_template('_result_page_.html')

if __name__ == '__main__':
    app.run(debug=True)