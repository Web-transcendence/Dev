import {getAvatar} from "./user.js"
import {activateBtn} from "./button.js";

export let connected = false;

window.addEventListener("popstate", () => {
    console.log("Navigating back:", window.location.pathname);
    loadPart(window.location.pathname);
});

document.addEventListener("DOMContentLoaded", async () => {
    // Constant button on the Single Page Application
    const aboutBtn = document.getElementById("about")!;
    const contactBtn = document.getElementById("contact")!;

    aboutBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/about"));
    contactBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/contact"));

    // For Client Connection
    if (await checkForTocken()) {
        getAvatar();
        handleConnection(true);
    }
    else
        handleConnection(false);
    const Ping = document.getElementById("pong");
    if (Ping)
        Ping.addEventListener("click", (event: MouseEvent) => navigate(event, "/pong"));
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
            console.log("token checker", error.message);
            return false;
        }
        return true;
    } catch (error) {
        console.error(error);
        return false;
    }
}




export function handleConnection(input: boolean) {
    const connect = document.getElementById('connect');
    const profile = document.getElementById('profile');
    if (input && profile && connect) {
        console.log("handleConnecton profile")
        connect.classList.add('hidden');
        profile.classList.remove('hidden');
        profile.addEventListener("click", (event: MouseEvent) => navigate(event, "/profile"));
    } else if (profile && connect) {
        localStorage.removeItem('token');
        localStorage.removeItem('avatar');
        console.log("handleConnecton connect")
        connect.classList.remove('hidden');
        profile.classList.add('hidden');
        connect.addEventListener("click", (event: MouseEvent) => navigate(event, "/connect"));
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
                if (reply.avatar) {
                    localStorage.setItem('avatar', reply.avatar);
                    console.log("reply.avatar");
                }
                if (reply.token)
                    localStorage.setItem('token', reply.token);
                loadPart("/connected");
                handleConnection(true);
            }
            console.log('Success:', reply);
        }
    }
    catch(error) {
        console.error('Error:', error);
    }
}

export async function navigate(event: MouseEvent, path: string): Promise<void> {
    handleConnection(await checkForTocken());
    console.log("Navigating back", path, connected);
    if (!connected && path == "/profile") {
        console.log("Got user");
        path = "/connect";
    }
    event.preventDefault();
    loadPart(path);
}

export async function loadPart(page: string): Promise<void> {
    window.history.pushState({}, "", page);
    try {
        await insert_tag(`part${page}`);
        console.log(`loadPart: ${page}`);
        activateBtn(page);
        getAvatar();
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
    console.log("PRORATA");
    afterInsert(url/*, container*/);
    newElement.innerHTML = html;
    container.appendChild(newElement);
}

function afterInsert(url: string,/* container: HTMLElement*/): void {
    console.log("afterInsert url :", url);
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
    if (url === "part/profile") {
        const changeAvatar = document.getElementById('avatarProfile') as HTMLImageElement;
        const avatar = localStorage.getItem('avatar');
        if (avatar && changeAvatar) {
            changeAvatar.src = avatar;
            console.log("Avatar Change Found");
        }
        if (!changeAvatar)
            console.log(" CHANge Avatar NOT FOUND AT ALLLLLLLLLLL");
        if (!avatar)
            console.log(" Avatar Avatar NOT FOUND AT ALLLLLLLLLLL");
    }
}


function activateGoogle(page: string) {
    console.log("activateGoogle", page);
    const container = document.getElementById('content') as HTMLElement;
    if (page === "/login" || page === "/connect") {
        const googleID = document.getElementById('googleidentityservice');
        if (!googleID) {
            console.log("afterInsert ADD googleID");
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
                console.log("afterInsert first REMOVE googlemeta");
                googlemeta.remove();
            }
            if (googleID) {
                console.log("afterInsert first REMOVE googleID");
                googleID.remove();
            }
            const googleIP = document.getElementById('googleidentityservice');
            if (!googleIP) {
                console.log("afterInsert just removed ADD googleID");
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
    if (googlemeta) {
        console.log("afterInsert second REMOVE googlemeta");
        googlemeta.remove();
    }
    if (googleID) {
        console.log("afterInsert second REMOVE googleID");
        googleID.remove();
    }
}


export function validateRegister(result: { nickName: string; email: string; password: string}): void {
    console.log(result);
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

