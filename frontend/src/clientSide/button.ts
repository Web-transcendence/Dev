import {addFriend, login, profile, register, setAvatar, verify2fa, joinTournament} from "./user.js";
import {init2fa} from "./user.js";
import {friendList} from "./friends.js";
import {handleConnection, navigate} from "./front.js";
// import {tdStop, TowerDefense} from "./td.js";
import { editProfile } from "./editInfoProfile.js";
import {  displayTournaments } from "./tournaments.js";
import {  DispayNotification } from "./notificationHandler.js";

const mapButton : {[key: string] : () => void} = {
    "/connect" : connectBtn,
    "/login": loginBtn,
    "/profile": profileBtn,
    "/logout": logoutBtn,
    "/factor" : factor,
    "/pongMode" : pongMode,
    "/towerMode" : towerMode,
    "/tournaments" : tournaments,
    "/lobby" : lobby
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

async function profileBtn() {
    const avatarImg = document.getElementById('avatarProfile') as HTMLImageElement
    const avatar = localStorage.getItem('avatar')
    if (avatar)
        avatarImg.src = avatar
    await profile();
    await friendList();
    editProfile();
    document.getElementById('logout')?.addEventListener("click", (event: MouseEvent) => navigate("/logout", event));
    const addFriendBtn = document.getElementById("friendNameBtn") as HTMLButtonElement;
    const addFriendIpt = document.getElementById("friendNameIpt") as HTMLButtonElement;
    if (addFriendBtn && addFriendIpt)
        addFriendBtn.addEventListener("click", async () => {
            await addFriend(addFriendIpt.value);
            await friendList();
        });
    const activeFA = localStorage.getItem('activeFA');
    if (activeFA) {
        document.getElementById('totalFactor')?.classList.add("hidden")
        document.getElementById('activeFactor')?.classList.remove("hidden")
    } else {
        const initFa = document.getElementById("initFa") as HTMLButtonElement;
        if (initFa) {
            initFa.addEventListener("click", async () => {
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

                    input.addEventListener("keydown", async (event: KeyboardEvent) => {
                        if (event.key === "Enter") {
                            await verify2fa(input.value)
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
            await verify2fa(input.value)
    })
}

function pongMode() {
    document.getElementById("pongRemote")?.addEventListener("click", (event: MouseEvent) => navigate("/pongRemote", event));
}

function towerMode() {
    document.getElementById("towerRemote")?.addEventListener("click", (event: MouseEvent) => navigate("/towerRemote", event));
}

// function tower() {
//     tdStop()
//     console.log("prout");
//     TowerDefense()
// }

function tournaments() {
    const tournaments : {id: number, name: string} []= [
        {id: 4, name: 'junior'},
        {id: 8, name: 'contender'},
        {id: 16, name: 'major'},
        {id: 32, name: 'worlds'}];
    for (const parse of tournaments)
        document.getElementById(`${parse.id}`)
            ?.addEventListener("click", async (event) => {
                await navigate('/lobby', event)
                sessionStorage.setItem('idTournaments', JSON.stringify(parse.id));
                sessionStorage.setItem('nameTournaments', JSON.stringify(parse.name));
            });
}

async function lobby() {
    await joinTournament()
    const id = sessionStorage.getItem('idTournaments');
    const name = sessionStorage.getItem('nameTournaments');
    if (!id || !name) {
        DispayNotification("Missing tournament information.");
        await navigate("/home");
        return ;
    }
    const toIntId = Number.parseInt(id);
    if (isNaN(toIntId)) DispayNotification("Invalid tournament ID.");
    await displayTournaments(toIntId, name);
}