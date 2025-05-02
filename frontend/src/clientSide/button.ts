import {addFriend, login, profile, register, setAvatar, verify2fa} from "./user.js";
import {init2fa} from "./user.js";
import {friendList} from "./friends.js";
import {connected, handleConnection, navigate} from "./front.js";


const mapButton : {[key: string] : () => void} = {
    "/connect" : connectBtn,
    "/login": loginBtn,
    "/profile": profileBtn,
    "/logout": logoutBtn,
    "/editProfile" : editProfileBtn,
    "/2fa" : factor,
    "/pongMode" : pongMode,
    "/towerMode" : towerMode
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
    const nickName = document.getElementById("profileNickName")!;
    const email = document.getElementById("profileEmail")!;
    if (email && nickName) {
        profile(nickName, email);
        friendList();
    }
    const logoutBtn = document.getElementById('logout')  as HTMLButtonElement;
    if (logoutBtn && connected)
        logoutBtn.addEventListener("click", (event: MouseEvent) => navigate("/logout", event));
    const addFriendBtn = document.getElementById("friendNameBtn") as HTMLButtonElement;
    const addFriendIpt = document.getElementById("friendNameIpt") as HTMLButtonElement;
    if (addFriendBtn && addFriendIpt) {
        addFriendBtn.addEventListener("click", () => addFriend(addFriendIpt.value));
        friendList();
    }
    document.getElementById('totalFactor')?.classList.add("hidden")
    document.getElementById('activeFactor')?.classList.remove("hidden")
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
    document.getElementById("inputAvatar")?.addEventListener("change", async (event: Event) => {
        const target = event.target as HTMLInputElement
        await setAvatar(target)
    });
}

function logoutBtn() {
    handleConnection(false);
    localStorage.removeItem('avatar');
    localStorage.removeItem('token');
    const avatar = document.getElementById("avatar") as HTMLImageElement;
    if (avatar)
        avatar.src = "../images/logout.png";
    const nickName = document.getElementById("nickName") as HTMLSpanElement;
    if (nickName)
        nickName.textContent = '';
}

function editProfileBtn() {
    document.getElementById('editAvatar')?.addEventListener("change", async (event: Event)=> {
            const target = event.target as HTMLInputElement;
            await setAvatar(target);
        }
    );
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
}

function towerMode() {
    document.getElementById("towerRemote")?.addEventListener("click", (event: MouseEvent) => navigate("/towerRemote", event));
}