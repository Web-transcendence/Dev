import {loadPart} from './insert.js';

declare const AOS: any

document.addEventListener("DOMContentLoaded", async () => {
    constantButton();
    AOS.init({
        once: true,
        duration: 800,
    })
    await loadPart(window.location.pathname || "/home");

    window.addEventListener("popstate", async () => {
        await loadPart(window.location.pathname || "/home");
    });
});

function constantButton() {
   // Logo
    document.getElementById('home')?.addEventListener("click", async (event: MouseEvent)=>
        navigate("/home", event));
   // Nav
    document.getElementById("about")?.addEventListener("click", (event: MouseEvent)=>
        navigate("/about", event));
    document.getElementById("contact")?.addEventListener("click", (event: MouseEvent)=>
        navigate("/contact", event));
    document.getElementById("shopDiscovery")?.addEventListener("click", (event: MouseEvent)=>
        navigate("/shopDiscovery", event));
    // Footer
    document.getElementById("mentionsLegales")?.addEventListener("click", (event: MouseEvent)=>
        navigate("/mentionsLegales", event));
    document.getElementById("PolitiqueConf")?.addEventListener("click", (event: MouseEvent)=>
        navigate("/PolitiqueConf", event));
    document.getElementById("CGU")?.addEventListener("click", (event: MouseEvent)=>
        navigate("/CGU", event));
    document.getElementById("mentionCopyright")?.addEventListener("click", (event: MouseEvent)=>
        navigate("/mentionCopyright", event));
}

export async function navigate(path: string, event?: MouseEvent) {
    event?.preventDefault();

    if (window.location.pathname === path) return;
    history.pushState({}, "", path);
    await loadPart(path);
    constantButton();
}
