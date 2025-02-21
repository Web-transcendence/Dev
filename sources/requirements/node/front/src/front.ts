
// Gérer la navigation lors de l'utilisation des boutons
document.addEventListener("DOMContentLoaded", () => {
    // Récupérer les boutons
    const aboutBtn = document.getElementById("about")!;
    const contactBtn = document.getElementById("contact")!;
    const registerBtn = document.getElementById("register")!;
    const loginBtn = document.getElementById("login")!;

    // Ajouter les écouteurs d'événements sur les boutons
    aboutBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/about"));
    contactBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/contact"));
    registerBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/register"));
    registerBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/login"));

    // Vous pouvez ajouter plus de boutons ici si nécessaire
    // Charger la page initiale en fonction de l'URL actuelle
    loadPage(window.location.pathname);
});

// Gérer la navigation via l'historique du navigateur (back/forward)
window.onpopstate = () => {
    loadPage(window.location.pathname)

};

// Fonction pour gérer la navigation sans recharger la page
function navigate(event: MouseEvent, path: string): void {
    event.preventDefault();
    window.history.pushState({}, "", path);  // Ajout de l'état dans l'historique
    loadPage(path);  // Chargement du contenu dynamique
}

// Fonction pour charger les pages dynamiquement
async function loadPage(page: string): Promise<void> {
    const content = document.getElementById("content") as HTMLElement;

    // Si aucune page n'est spécifiée, afficher un message par défaut
    if (!page || page === "/") {
        content.innerHTML = "<p>Veuillez choisir une option pour charger du contenu.</p>";
        return;
    }

    // Tentative de récupération de la page dynamique
    try {
        const res = await fetch(page);
        if (!res.ok) throw new Error("Page non trouvée");

        const html = await res.text();  // Récupérer le contenu HTML de la page
        content.innerHTML = html;  // Injecter le contenu dans le div #content
    } catch (error) {
        console.error(error);
        content.innerHTML = "<h2>404 - Page non trouvée</h2>";
    }
}
