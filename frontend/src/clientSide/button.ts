import {addFriend, getAvatar, getFriendList, login, profile, register, setAvatar, verify2fa} from "./user.js";
import { friendList, init2fa} from "./user.js";
import {connected, handleConnection, navigate} from "./front.js";


const mapButton : {[key: string] : () => void} = {
    "/connect" : connectBtn,
    "/login": loginBtn,
    "/profile": profileBtn,
    "/logout": logoutBtn,
    "/editProfile" : editProfileBtn
}

export function activateBtn(page: string) {
    document.getElementById("pongConnected")?.addEventListener("click", (event: MouseEvent) => navigate(event, "/pong"));
    document.getElementById('home')?.addEventListener("click", (event: MouseEvent) => navigate(event, "/home"));
    document.getElementById("loginButton")?.addEventListener("click", (event: MouseEvent) => navigate(event, "/login"));
    const logoutBtn = document.getElementById('logout')  as HTMLButtonElement;
    if (logoutBtn && connected)
        logoutBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/logout"));
    if (page in mapButton)
        mapButton[page]();
    //navigation page
    const pongMode = document.getElementById("pongMode") as HTMLButtonElement;
    if (pongMode)
        pongMode.addEventListener("click", (event: MouseEvent) => navigate(event, "/pongMode"));
    const tower = document.getElementById("tower") as HTMLButtonElement;
    if (tower)
        tower.addEventListener("click", (event: MouseEvent) => navigate(event, "/tower"));
    const tournaments = document.getElementById("tournaments") as HTMLButtonElement;
    if (tournaments)
        tournaments.addEventListener("click", (event: MouseEvent) => navigate(event, "/tournaments"));
    document.getElementById('editAvatar')?.addEventListener("change", async (event: Event)=> {
        const target = event.target as HTMLInputElement;
        await setAvatar(target);
        }
    );

}

function connectBtn() {
    const button = document.getElementById("registerButton") as HTMLButtonElement;
    if (button)
        register(button);
}

function loginBtn() {
    document.getElementById('connectPageBtn')?.addEventListener("click", (event: MouseEvent) => navigate(event, "/connect"));
    const button = document.getElementById("loginButton") as HTMLButtonElement;
    if (button)
        login(button);
}

function profileBtn() {
    const nickName = document.getElementById("profileNickName")!;
    const email = document.getElementById("profileEmail")!;
    // const container = document.getElementById('content') as HTMLElement;
    if (email && nickName) {
        profile(/*container,*/ nickName, email);
        friendList();
    }
    const editProfileBtn = document.getElementById("editProfileButton") as HTMLButtonElement;
    const addFriendBtn = document.getElementById("friendNameBtn") as HTMLButtonElement;
    const addFriendIpt = document.getElementById("friendNameIpt") as HTMLButtonElement;
    if (addFriendBtn && addFriendIpt) {
        console.log("addFriendIptaddFriendIpt");
        addFriendBtn.addEventListener("click", () => addFriend(addFriendIpt.value));
        friendList();
    }
    const initfa = document.getElementById("initfa") as HTMLButtonElement;
    if (initfa) {
        console.log("initfa");
        initfa.addEventListener("click", async () => {
            const qrcode = await init2fa();
            if (qrcode == undefined) {
                console.log("ErrorDisplay: qrcode not found!");
                return;
            }
            console.log("2FA initialized:", qrcode);
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
    document.getElementById("profilePicture")?.addEventListener("change", async (event: Event) => {
        const target = event.target as HTMLInputElement
        await setAvatar(target)
    });

    if (editProfileBtn) {
        editProfileBtn.addEventListener("click", async () => console.log(await getAvatar()));
    }
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

}