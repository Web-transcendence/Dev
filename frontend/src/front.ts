// Gérer la navigation lors de l'utilisation des boutons
document.addEventListener("DOMContentLoaded", () => {
    // Récupérer les boutons
    const aboutBtn = document.getElementById("about")!;
    const contactBtn = document.getElementById("contact")!;
    const registerBtn = document.getElementById("register")!;

    // Ajouter les écouteurs d'événements sur les boutons
    aboutBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/about"));
    contactBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/contact"));
    registerBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/register"));

    // Charger la page initiale en fonction de l'URL actuelle
    loadPart(window.location.pathname);
});

// Gérer la navigation via l'historique du navigateur (back/forward)
window.onpopstate = () => {
    loadPart(window.location.pathname);
};

// Fonction pour gérer la navigation sans recharger la page
function navigate(event: MouseEvent, path: string): void {
    event.preventDefault();
    window.history.pushState({}, "", path);  // Ajout de l'état dans l'historique
    loadPart(path);  // Chargement du contenu dynamique
}

// Fonction pour charger les pages dynamiquement
async function loadPart(page: string): Promise<void> {
    const container = document.getElementById('content') as HTMLElement;
    container.innerHTML = '';
    if (!page || page === "/") { // Si aucune page n'est spécifiée, afficher un message par défaut
        container.innerHTML = "<p>Choisissez une option pour charger le contenu.</p>";
        return;
    }
    try { // Tentative de récupération de la page dynamique
        if (!document.body) console.log("===========body is null");
            const res = await fetch(`part${page}`);
        const newElement = document.createElement('div');
        newElement.className = 'balise';
        if (!res.ok) throw new Error("Page non trouvée");
        const html = await res.text();  // Récupérer le contenu HTML de la page
        newElement.innerHTML = html;  // Injecter le contenu dans le div #content
        container.appendChild(newElement);
        if (page === "/register") {
            const button = document.getElementById("registerButton")!;
            if (button) { // Enable the button
                // button.disabled = false;
                button.addEventListener("click", async (event) => {
                    const myForm = document.getElementById("myForm") as HTMLFormElement;
                    const formData = new FormData(myForm);
                    const data: Record<string, unknown> = Object.fromEntries(formData as unknown as Iterable<readonly any[]>);

                    await fetch('http://api-gateway:8000/user-management/sign-up', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(data)
                    });
                });
            }
        }
    } catch (error) {
        console.error(error);
        container.innerHTML = "<h2>404 - Page non trouvée</h2>";
    }
}

