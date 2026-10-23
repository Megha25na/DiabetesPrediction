const form = document.getElementById("form");
const result = document.getElementById("result");

form.addEventListener("submit", async function (event) {
  event.preventDefault();   // stop the page from reloading
  result.className = "";
  result.innerText = "Predicting...";

  // Collect the values from the form
  const patient = {
    age: document.getElementById("age").value,
    gender: document.getElementById("gender").value,
    hypertension: document.getElementById("hypertension").value,
    heart_disease: document.getElementById("heart_disease").value,
    smoking_history: document.getElementById("smoking_history").value,
    bmi: document.getElementById("bmi").value,
    HbA1c_level: document.getElementById("hba1c").value,
    blood_glucose_level: document.getElementById("glucose").value
  };

  try {
    // Send the values to the Python models
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patient)
    });
    const data = await response.json();

    result.innerText = "";

    if (data.error) {
      result.className = "red";
      result.innerText = data.error;
      return;
    }

    // Show one line for each model
    data.results.forEach(function (r) {
      const line = document.createElement("div");
      line.className = "line " + (r.prediction === 1 ? "red" : "green");
      const label = r.prediction === 1 ? "Diabetic" : "Not Diabetic";
      line.innerText = r.name + ": " + label + " (" + r.probability + "% chance of diabetes)";
      result.appendChild(line);
    });
  } catch (error) {
    result.className = "red";
    result.innerText = "Could not reach the server. Please try again in a minute.";
  }
});
