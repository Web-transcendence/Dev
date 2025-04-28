import {getAvatar} from "./user.js"
import {activateBtn} from "./button.js";

export let connected = false;

window.addEventListener("popstate", () => {
    loadPart(window.location.pathname);
});

document.addEventListener("DOMContentLoaded", async () => {
    constantButton(); // Constant button on the Single Page Application
    // For Client Connection
    if (await checkForTocken()) {
        getAvatar();
        handleConnection(true);
    }
    else
        handleConnection(false);
    const towerDefense = document.getElementById("towerDefense");
    if (towerDefense) {
        console.log("Tower Defense:");
        towerDefense.addEventListener("click", (event: MouseEvent) => navigate(event, "/towerDefense"));
    }
    loadPart("/home");
});

async function checkForTocken(): Promise<boolean>  {
    try {
        const token = localStorage.getItem('token');
        const response = await fetch('http://localhost:3000/authJWT', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'authorization': 'Bearer ' + token,
            },
        });
        if (!response.ok) {
            const error = await response.json();
            return false;
        }
        return true;
    } catch (error) {
        console.error(error);
        return false;
    }
}

function constantButton() {
    //Duo Button
    document.getElementById('connect')?.addEventListener("click", (event: MouseEvent) => navigate(event, "/connect"));
    document.getElementById('profile')?.addEventListener("click", (event: MouseEvent) => navigate(event, "/profile"));
    //navigation page
    document.getElementById('home')?.addEventListener("click", (event: MouseEvent) => navigate(event, "/home"));
    document.getElementById("pongMode")?.addEventListener("click", (event: MouseEvent) => navigate(event, "/pongMode"));
    document.getElementById("tower")?.addEventListener("click", (event: MouseEvent) => navigate(event, "/tower"));
    document.getElementById("tournaments")?.addEventListener("click", (event: MouseEvent) => navigate(event, "/tournaments"));
    // Footer
    document.getElementById("about")?.addEventListener("click", (event: MouseEvent) => navigate(event, "/about"));
    document.getElementById("contact")?.addEventListener("click", (event: MouseEvent) => navigate(event, "/contact"));
    //Pong
    document.getElementById("pong")?.addEventListener("click", (event: MouseEvent) => navigate(event, "/pong"));
}


export function handleConnection(input: boolean) {
    if (input) {
        document.getElementById('connect')?.classList.add('hidden');
        document.getElementById('profile')?.classList.remove('hidden');
    } else {
        localStorage.removeItem('token');
        localStorage.removeItem('avatar');
        localStorage.removeItem('nickName');
        localStorage.removeItem('factor');
        document.getElementById('connect')?.classList.remove('hidden');
        document.getElementById('profile')?.classList.add('hidden');
    }
    connected = input;
}

// @ts-ignore
window.CredentialResponse = async (credit: { credential: string }) => {
    try {
        const response = await fetch('http://localhost:3000/user-management/auth/google', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({credential: credit.credential})
        });
        if (!response.ok)
            console.error('Error: From UserManager returned an error');
        else {
            const reply = await response.json();
            if (reply.valid) {
                if (reply.avatar)
                    // setAvatar(reply.avatar);
                if (reply.token)
                    localStorage.setItem('token', reply.token)
                if (reply.nickName)
                    localStorage.setItem('nickName', reply.nickName)
                loadPart("/connected");
                handleConnection(true);
                getAvatar();
            }
        }
    }
    catch(error) {
        console.error('Error:', error);
    }
}

export async function navigate(event: MouseEvent, path: string): Promise<void> {
    handleConnection(await checkForTocken());
    if (!connected && path == "/profile") {
        path = "/connect";
    }
    event.preventDefault();
    loadPart(path);
}

export async function loadPart(page: string): Promise<void> {
    window.history.pushState({}, "", page);
    try {
        await insert_tag(`part${page}`);
        activateBtn(page);
        activateGoogle(page);
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

async function insert_tag(url: string): Promise<void>{
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
    afterInsert(url);
    newElement.innerHTML = html;
    container.appendChild(newElement);
}

function afterInsert(url: string): void {
    if (url === "part/pong") {
        if (!document.querySelector('script[src="/static/dist/pong.js"]')) {
            const script = document.createElement('script');
            script.src = "/static/dist/pong.js";
            document.body.appendChild(script);
        }
    } else {
        const existingScript = document.querySelector('script[src="/static/dist/pong.js"]');
        if (existingScript)
            existingScript.remove();
    }
    if (url === "part/towerDefense") {
        if (!document.querySelector('script[src="/static/dist/td.js"]')) {
            const script = document.createElement('script');
            script.src = "/static/dist/td.js";
            document.body.appendChild(script);
        }
    } else {
        const existingScript = document.querySelector('script[src="/static/dist/td.js"]');
        if (existingScript)
            existingScript.remove();
    }
}


function activateGoogle(page: string) {
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


export function validateRegister(result: { nickName: string; email: string; password: string}): void {
    const nickNameErrorMin = document.getElementById("nickNameError") as HTMLSpanElement;
    const emailError = document.getElementById("emailError") as HTMLSpanElement;
    const passwordError = document.getElementById("passwordError") as HTMLSpanElement;
    if (result.nickName)
        nickNameErrorMin.classList.remove("hidden");
    else {
        if (!nickNameErrorMin.classList.contains("hidden")) {
            nickNameErrorMin.classList.add("hidden");
        }
    }
    if (result.email)
        emailError.classList.remove("hidden");
    else {
        if (!emailError.classList.contains("hidden")) {
            emailError.classList.add("hidden");
        }
    }
    if (result.password)
        passwordError.classList.remove("hidden");
    else {
        if (!passwordError.classList.contains("hidden")) {
            passwordError.classList.add("hidden");
        }
    }
}

