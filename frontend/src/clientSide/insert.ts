import {activateBtn} from "./button.js";

export async function loadPart(page: string) {
    window.history.pushState({}, "", page);
    try {
        await insertTag(`part${page}`);
        activateBtn(page);
        activateGoogle(page);
        insertScript(page);
    } catch (error) {
        console.error(error);
        const container = document.getElementById('content') as HTMLElement;
        container.innerHTML = '';
        container.innerHTML = `<div class="bg-gray-900 text-white font-mono flex items-center justify-center min-h-screen">
        <div class="text-center space-y-6">
        <span class="block text-9xl text-pink-500">404 - NOT FOUND</span>
        <p class="text-5xl leading-relaxed">Oops! This page does not exist.</p>
        </div>
        </div>`;
    }
}

export async function insertTag(url: string): Promise<void>{
    const container = document.getElementById('content') as HTMLElement;
    if (url === "part/pong") {
        const existingScript = document.querySelector('script[src="/static/dist/pong.js"]');
        if (existingScript)
            return ;
    }
    const res = await fetch(url);
    const newElement = document.createElement('div');
    newElement.className = 'tag';
    if (!res.ok)
        throw Error("Page not found: element missing.");
    const html = await res.text();
    if (container.innerHTML.includes(html))
        return;
    if (html.includes(container.innerHTML))
        return;
    container.innerHTML = '';
    newElement.innerHTML = html;
    container.appendChild(newElement);
}


export function activateGoogle(page: string) {
    const container = document.getElementById('content') as HTMLElement;
    if (page === "/login" || page === "/connect") {
        const googleID = document.getElementById('googleidentityservice');
        if (!googleID) {
            const script = document.createElement('script');
            script.src = "https://accounts.google.com/gsi/client";
            script.async = true;
            script.defer = true;
            container.appendChild(script);
        }
        else {
            const googleID = document.getElementById('googleidentityservice');
            const googlemeta = document.querySelector('meta[http-equiv="origin-trial"]');
            if (googlemeta) {
                googlemeta.remove();
            }
            if (googleID) {
                googleID.remove();
            }
            const googleIP = document.getElementById('googleidentityservice');
            if (!googleIP) {
                const script = document.createElement('script');
                script.src = "https://accounts.google.com/gsi/client";
                script.async = true;
                script.defer = true;
                container.appendChild(script);
            }
        }
    }
    const googleID = document.getElementById('googleidentityservice');
    const googlemeta = document.querySelector('meta[http-equiv="origin-trial"]');
    if (googlemeta)
        googlemeta.remove();
    if (googleID)
        googleID.remove();
}

export function insertScript(page: string): void {
    const scripts: Record<string, string> = {
        "/pong": "/static/dist/pong.js",
        "/towerDefense": "/static/dist/td.js",
    };

    const currentScriptSrc = scripts[page];

    // Supprime tous les scripts sauf celui qui correspond à la page actuelle
    Object.entries(scripts).forEach(([_, src]) => {
        if (src !== currentScriptSrc) {
            const existing = document.querySelector<HTMLScriptElement>(`script[src="${src}"]`);
            if (existing) existing.remove();
        }
    });

    // Si nécessaire, insère le script correspondant
    if (currentScriptSrc && !document.querySelector(`script[src="${currentScriptSrc}"]`)) {
        const script = document.createElement('script');
        script.src = currentScriptSrc;
        document.body.appendChild(script);
    }
}
