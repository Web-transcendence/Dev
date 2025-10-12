import { displayNotification } from "./notificationHandler.js";
import { openModal} from './modal.js'
import {navigate} from './front.js'

const mapButton : {[key: string] : () => void} = {
    "/contact" : contact,
    "/about" : about,
    "/home" : home,
    "/shopDiscovery" : shopDiscovery,
    "/mentionsLegales" : mentionsLegales,
}

export function activateBtn(page: string) {
    if (page in mapButton)
        mapButton[page]();
}

async function contact() {
    let ContactEmail: string | null = null;
    let ContactMessage: string | null = null;
    document.getElementById("contactBtn")?.addEventListener("click", async (event: MouseEvent) => {
        event.preventDefault();

        const emailInput = document.getElementById("email") as HTMLInputElement | null;
        const messageInput = document.getElementById("message") as HTMLTextAreaElement | null;

        if (emailInput && messageInput) {
            const email = emailInput.value.trim();
            const message = messageInput.value.trim();

            if (email && message) {
                ContactEmail = email;
                ContactMessage = message;
                try {
                    const res = await fetch("/api/get-message", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({email: ContactEmail, message: ContactMessage}),
                    });

                    const data = await res.json();
                    if (res.ok) {
                        displayNotification(`Merci ! Message envoyé`);
                        emailInput.value = "";
                        messageInput.value = "";
                    } else if (res.status === 429) {
                        displayNotification("Trop de tentatives, réessayez dans une minute !");
                    } else {
                        displayNotification(`${data.error}`);
                    }
                } catch (err) {
                    displayNotification(`Impossible de contacter le serveur.`);
                    console.error(err);
                }
            } else {
                displayNotification("Veuillez entrer une adresse email valide.");
            }
            emailInput.value = "";
            messageInput.value = "";
        } else {
            displayNotification("Merci de remplir tous les champs !");
        }
    });
}


async function about() {

    document.getElementById("beginCustomer")?.addEventListener(
        "click", (event: MouseEvent) => openModal());
}

async function home() {
    document.getElementById("joinHome")?.addEventListener(
        "click", (event: MouseEvent) => openModal());
}

async function shopDiscovery() {
    document.getElementById("beginShop")?.addEventListener(
        "click", (event: MouseEvent) => openModal());
}

async function mentionsLegales() {
    document.getElementById("contactml")?.addEventListener(
        "click", (event: MouseEvent) => navigate("/contact", event));
    document.getElementById("contactml2")?.addEventListener(
        "click", (event: MouseEvent) => navigate("/contact", event));
    document.getElementById("contactml3")?.addEventListener(
        "click", (event: MouseEvent) => navigate("/contact", event));
    document.getElementById("contactml4")?.addEventListener(
        "click", (event: MouseEvent) => navigate("/contact", event));
}