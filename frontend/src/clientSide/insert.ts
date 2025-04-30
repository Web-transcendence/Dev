import {activateBtn} from "./button.js";

export async function loadPart(page: string) {
    try {
        await insertTag(`part${page}`);
        insertScript(page);
        activateBtn(page);
        activateGoogle(page);
    } catch (error) {
        console.error(error);
        const container = document.getElementById('content') as HTMLElement;
        container.innerHTML = '';
        document.getElementById('notFound')?.classList.remove('hidden');
    }
}

export async function insertTag(url: string): Promise<void>{
    const container = document.getElementById('content') as HTMLElement;
    const res = await fetch(url);
    const newElement = document.createElement('div');
    newElement.className = 'tag';
    if (!res.ok)
        throw Error("Page not found: element missing.");
    const html = await res.text();
    if (container.innerHTML.includes(html)) {
        console.log("sortie 1")
        return;
    }
    document.getElementById('notFound')?.classList.add('hidden');
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
        "/pongRemote": "/static/dist/pong.js",
        "/towerRemote": "/static/dist/td.js",
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
