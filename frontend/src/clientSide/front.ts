import {getAvatar} from './user.js'
import {loadPart} from './insert.js';

export let connected = false;

window.addEventListener("popstate", () => {
    console.log("popstate");
    loadPart(window.location.pathname)
});

document.addEventListener("DOMContentLoaded", async () => {
    constantButton(); // Constant button on the Single Page Application
    // For Client Connection
    if (await checkForToken()) {
        await getAvatar();
        handleConnection(true);
    }
    else
        handleConnection(false);
    const path = localStorage.getItem('path');
    if (path) loadPart(path)
    else loadPart("/home")
});

async function checkForToken(): Promise<boolean>  {
    try {
        const token = localStorage.getItem('token');
        const response = await fetch(`/authJWT`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'authorization': 'Bearer ' + token,
            },
        });
        if (!response.ok) {
            const error = await response.json();
            console.error(error)
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
    document.getElementById('avatar')?.addEventListener('click', () => {
        if (connected)
            document.getElementById('connect')?.click();
        else
            document.getElementById('profile')?.click();
    });
    //navigation page
    document.getElementById('home')?.addEventListener("click", (event: MouseEvent) => navigate(event, "/home"));
    document.getElementById("pongMode")?.addEventListener("click", (event: MouseEvent) => navigate(event, "/pongMode"));
    document.getElementById("towerDefense")?.addEventListener("click", (event: MouseEvent) => navigate(event, "/tower"));
    document.getElementById("tournaments")?.addEventListener("click", (event: MouseEvent) => navigate(event, "/tournaments"));
    // Footer
    document.getElementById("about")?.addEventListener("click", (event: MouseEvent) => navigate(event, "/about"));
    document.getElementById("contact")?.addEventListener("click", (event: MouseEvent) => navigate(event, "/contact"));
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
        const response = await fetch(`/user-management/auth/google`, {
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
                await loadPart("/connected");
                handleConnection(true);
                await getAvatar();
            }
        }
    }
    catch(error) {
        console.error('Error:', error);
    }
}

export async function navigate(event: MouseEvent, path: string): Promise<void> {
    handleConnection(await checkForToken());
    if (!connected && path == "/profile") {
        path = "/connect";
    }
    event.preventDefault();
    await loadPart(path);
}
