import {getAvatar} from './user.js'
import {loadPart} from './insert.js';
import {sseConnection} from "./serverSentEvent.js";
import {joinTournament, quitTournaments, getBrackets} from "./tournaments.js";
import {displayNotification} from "./notificationHandler.js";

declare const tsParticles: any;
declare const AOS: any;

export let connected = false;

window.addEventListener("popstate", (event) => {
    if ((window.location.pathname === '/connect' || window.location.pathname === '/login') && connected ) {
        history.replaceState(null, '', '/home');
        loadPart('/home')
    }
    else
        loadPart(window.location.pathname)
});

document.addEventListener("DOMContentLoaded", async () => {
    constantButton(); // Constant button on the Single Page Application
    // animate slides on scroll
    AOS.init({
        once: true,
        duration: 800,
    });
    // Reconnect User
    const token = sessionStorage.getItem("token");
    if (token && await checkForToken()) {
        await getAvatar();
        handleConnection(true);
    }
    else
        handleConnection(false);
    // For Client Connection
    document.getElementById('avatar')?.addEventListener('click', () => {
        if (connected)
            document.getElementById('profile')?.click();
        else
            document.getElementById('connect')?.click();
    });
    const tournamentId = sessionStorage.getItem('idTournaments')
    if (tournamentId)
        await joinTournament(Number(tournamentId))
    const path = sessionStorage.getItem('path');
    if (path && !(!connected && path === '/profile'))
        await loadPart(path)
    else
        await loadPart("/home")
    await sseConnection()
});

tsParticles.load("tsparticles", {
    fullScreen: { enable: false },
    particles: {
        number: { value: 100 },
        size: { value: 6 },
        move: { enable: true, speed: 1 },
        opacity: { value: 0.5 },
        color: { value: "#ffffff" },
    },
    background: {
        color: "#000000"
    }
});

async function checkForToken(): Promise<boolean>  {
    try {
        const token = sessionStorage.getItem('token');
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
    document.getElementById('connect')?.addEventListener("click", (event: MouseEvent) => navigate("/connect", event));
    document.getElementById('profile')?.addEventListener("click", (event: MouseEvent) => navigate("/profile", event));
    //navigation page
    document.getElementById('home')?.addEventListener("click", async (event: MouseEvent) => {
        await navigate("/brackets", event)
        await getBrackets(4);
    });
    document.getElementById("pongMode")?.addEventListener("click", (event: MouseEvent) => navigate("/pongMode", event));
    document.getElementById("towerDefense")?.addEventListener("click", (event: MouseEvent) => navigate("/towerMode", event));
    document.getElementById("tournaments")?.addEventListener("click", (event: MouseEvent) => navigate("/tournaments", event));
    document.getElementById("matchHistory")?.addEventListener("click", (event: MouseEvent) => navigate("/matchHistory", event));
    // Footer
    document.getElementById("about")?.addEventListener("click", (event: MouseEvent) => navigate("/about", event));
    document.getElementById("contact")?.addEventListener("click", (event: MouseEvent) => navigate("/contact", event));
}


export function handleConnection(input: boolean) {
    if (input) {
        document.getElementById('connect')?.classList.add('hidden');
        document.getElementById('profile')?.classList.remove('hidden');
    } else {
        
        sessionStorage.clear();
        sessionStorage.clear();
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
                    sessionStorage.setItem('avatar', reply.avatar)
                if (reply.token) {
                    console.log('VALID RESPONSE', reply.token);
                    sessionStorage.setItem('token', reply.token)
                }
                console.log('ID GOOGLE ', reply.id)
                if (reply.id)
                    sessionStorage.setItem('id', reply.id)
                if (reply.nickName)
                    sessionStorage.setItem('nickName', reply.nickName)
                navigate('/connected');
                await getAvatar();
                await sseConnection()
            }
        }
    }
    catch(error) {
        console.error('Error:', error);
    }
}

export async function navigate(path: string, event?: MouseEvent): Promise<void> {
    if (event) event.preventDefault();

    handleConnection(await checkForToken());
    if (!connected && path == "/profile")
        path = "/connect";
    if (connected && path == "/connect")
        path = "/profile";
    history.pushState({}, "", path);
    console.log("pushState :", path);
    const idT = sessionStorage.getItem('idTournaments')
    if (idT && path != '/lobby') {
        displayNotification(`You left the Tournament`);
        sessionStorage.removeItem('idTournaments');
        sessionStorage.removeItem('nameTournaments');
        await quitTournaments()
    }
    await loadPart(path);
}
