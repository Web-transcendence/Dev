document.addEventListener("DOMContentLoaded", () => {
    const button = document.getElementById("myButton");
    if (button) {
        button.addEventListener("click", getTest);
    }
});

async function getTest() {
    try {
        const response = await fetch("http://localhost:8001/user-management/5");
        if (!response.ok) {
            throw new Error(`Request failed with status code ${response.status}`);
        }
        else {
            const test = await response.json();
            console.log(test);

            const resultDiv = document.getElementById("result");

            if (resultDiv) {
                resultDiv.innerHTML = `
                    <h3>Détails de l'utilisateur :</h3>
                    <p>ID: ${test.id}</p>
                    <p>Nom: ${test.name}</p>
               `;
            }
            else {
                console.log("nuul");
            }
        }

    } catch (err) {
        console.log(err);
        const resultDiv = document.getElementById("result");
        if (resultDiv) {
            resultDiv.innerHTML = '<p>Une erreur est survenue lors de la récupération des données.</p>';
        }
    }
}