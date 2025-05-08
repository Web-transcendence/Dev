import {addFriend, login, profile, register, setAvatar, setNickName, setPassword, verify2fa} from "./user.js";
import {init2fa} from "./user.js";
import {friendList} from "./friends.js";
import {handleConnection, navigate} from "./front.js";
import {tdStop, TowerDefense} from "./td.js";
import { editProfile } from "./editInfoProfile.js";
import {DispayNotification} from "./notificationHandler.js";
import {Pong} from "./pong.js";


const mapButton : {[key: string] : () => void} = {
    "/connect" : connectBtn,
    "/login": loginBtn,
    "/profile": profileBtn,
    "/logout": logoutBtn,
    "/factor" : factor,
    "/towerMode" : towerMode,
    "/towerRemote" : towerRemote,
    "/pongMode" : pongMode,
    "/pongRemote" : pongRemote,
    "/pongLocal" : pongLocal,
    "/pongWatch" : pongWatch
}

export function activateBtn(page: string) {
    if (page in mapButton)
        mapButton[page]();
}

function connectBtn() {
    document.getElementById("loginButton")?.addEventListener("click", (event: MouseEvent) => navigate("/login", event));
    const button = document.getElementById("registerButton") as HTMLButtonElement;
    if (button)
        register(button);
}

function loginBtn() {
    document.getElementById('connectPageBtn')?.addEventListener("click", (event: MouseEvent) => navigate("/connect", event));
    const button = document.getElementById("loginButton") as HTMLButtonElement;
    if (button)
        login(button)
}

function profileBtn() {
    const avatarImg = document.getElementById('avatarProfile') as HTMLImageElement
    const avatar = localStorage.getItem('avatar')
    if (avatar)
        avatarImg.src = avatar
    profile();
    friendList();
    document.getElementById("editProfileButton")?.addEventListener("click", () => {
        const nickInput = document.getElementById("profileNickName") as HTMLInputElement | null;
        const emailInput = document.getElementById("profileEmail") as HTMLInputElement | null;
        if (nickInput && emailInput) {
            const newNickName = nickInput.value.trim();
            const newEmail = emailInput.value.trim();
            console.log("New nickname:", newNickName);
            console.log("New email:", newEmail);
            setNickName(newNickName);
            // setPassword();
            // setEmail();
        }
    });
    document.getElementById('logout')?.addEventListener("click", (event: MouseEvent) => navigate("/logout", event));
    const addFriendBtn = document.getElementById("friendNameBtn") as HTMLButtonElement;
    const addFriendIpt = document.getElementById("friendNameIpt") as HTMLButtonElement;
    if (addFriendBtn && addFriendIpt)
        addFriendBtn.addEventListener("click", () => {
            addFriend(addFriendIpt.value);
            friendList();
        });
    const activeFA = localStorage.getItem('activeFA');
    if (activeFA) {
        document.getElementById('totalFactor')?.classList.add("hidden")
        document.getElementById('activeFactor')?.classList.remove("hidden")
    } else {
        const initfa = document.getElementById("initfa") as HTMLButtonElement;
        if (initfa) {
            initfa.addEventListener("click", async () => {
                const qrcode = await init2fa();
                if (qrcode == undefined) {
                    console.log("ErrorDisplay: qrcode not found!");
                    return;
                }
                const insertQrcode = document.getElementById("insertQrcode");
                if (insertQrcode) {
                    const img = document.createElement("img");
                    img.src = qrcode;
                    img.classList.add("h-3/4", "w-3/4", "p-4", "rounded-lg");
                    insertQrcode.innerHTML = "";
                    insertQrcode.appendChild(img);
                    const label = document.getElementById("codeFaInput");
                    if (label)
                        label.classList.remove("sr-only");
                    const input = document.getElementById("inputVerify") as HTMLInputElement;

                    input.addEventListener("keydown", async (event :KeyboardEvent) => {
                        if (event.key === "Enter") {
                            verify2fa(input.value)
                        }
                    })
                }
            });
        }
    }
    document.getElementById("inputAvatar")?.addEventListener("change", async (event: Event) => {
        const target = event.target as HTMLInputElement
        await setAvatar(target)
    });
}

function logoutBtn() {
    handleConnection(false);
    const avatar = document.getElementById("avatar") as HTMLImageElement;
    if (avatar) avatar.src = "../images/logout.png";
    const nickName = document.getElementById("nickName") as HTMLSpanElement;
    if (nickName) nickName.textContent = '';

}

function factor() {
    const input = document.getElementById("inputVerify") as HTMLInputElement;

    input.addEventListener("keydown", async (event :KeyboardEvent) => {
        if (event.key === "Enter")
            verify2fa(input.value)
    })
}

function pongMode() {
    document.getElementById("pongRemote")?.addEventListener("click", (event: MouseEvent) => navigate("/pongRemote", event));
    document.getElementById("pongLocal")?.addEventListener("click", (event: MouseEvent) => navigate("/pongLocal", event));
    document.getElementById("pongWatch")?.addEventListener("click", (event: MouseEvent) => navigate("/pongWatch", event));
}

function towerMode() {
    document.getElementById("towerRemote")?.addEventListener("click", (event: MouseEvent) => navigate("/towerRemote", event));
}

function towerRemote() {
    tdStop()
    TowerDefense()
}

function pongLocal() {
    Pong("local")
}

function pongRemote() {
    Pong("remote")
}

function pongWatch() {
    Pong("spec")
}