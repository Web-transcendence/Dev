
// Gérer la navigation lors de l'utilisation des boutons
document.addEventListener("DOMContentLoaded", () => {

    // const navigationEntry = performance.getEntriesByType('navigation')[0];
    // if (navigationEntry instanceof PerformanceNavigationTiming) {
    //     // Now TypeScript knows that navigationEntry is a PerformanceNavigationTiming object
    //     const navigationType = navigationEntry.type;
    //     if (navigationType === 'reload') {
    //         const page = window.location.pathname;
    //         loadPage("/");
    //         console.log('Page refreshed');
    //         // window.location.href = "/"; // Navigate to the home page
    //         reloadPage(page);
    //         return;
    //     } else {
    //         // console.log('Page loaded');
    //     }
    // }
    console.log("========22222222============");
    // Récupérer les boutons
    const aboutBtn = document.getElementById("about")!;
    const contactBtn = document.getElementById("contact")!;
    const registerBtn = document.getElementById("register")!;
    const loginBtn = document.getElementById("login")!;
    // Ajouter les écouteurs d'événements sur les boutons
    aboutBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/about"));
    contactBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/contact"));
    registerBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/register"));
    loginBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/login"));
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

async function reloadPage(part_name: string): Promise<void> {
    const content = await document.getElementById("content") as HTMLElement;
    try {
        if (part_name === "/") {
            console.log("====1======page is /");
            return;
        }
        if (!document.body) console.log("====2======body is null");
        console.log("====3======page is " + part_name);
        const res = await fetch(part_name);
        if (!res.ok) throw new Error("Page non trouvée");

        const html = await res.text();  // Récupérer le contenu HTML de la page
        console.log(html);
        content.innerHTML = html;  // Injecter le contenu dans le div #content
    } catch (error) {
        console.error(error);
        content.innerHTML = "<h2>404 - Page non trouvée</h2>";
    }
}

// Fonction pour charger les pages dynamiquement
async function loadPart(page: string): Promise<void> {
    const content = document.getElementById("content") as HTMLElement;
    // Si aucune page n'est spécifiée, afficher un message par défaut
    if (!page || page === "/") {
        content.innerHTML = "<p>Choisissez une option pour charger le contenu.</p>";
        return;
    }
    // Tentative de récupération de la page dynamique
    try {
        if (!document.body) console.log("===========body is null");
            const res = await fetch(`part${page}`);
        if (!res.ok) throw new Error("Page non trouvée");
        console.log("====3======page is " + page);
        const html = await res.text();  // Récupérer le contenu HTML de la page
        console.log(html);
        content.innerHTML = html;  // Injecter le contenu dans le div #content
    } catch (error) {
        console.error(error);
        content.innerHTML = "<h2>404 - Page non trouvée</h2>";
    }
}
