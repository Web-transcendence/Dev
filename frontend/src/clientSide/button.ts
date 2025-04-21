import {addFriend, getFriendList, login, profile, register} from "./user.js";
import {connected, handleConnection, navigate} from "./front.js";


const mapButton : {[key: string] : () => void} = {
    "/connect" : connectBtn,
    "/login": loginBtn,
    "/profile": profileBtn,
    "/logout": logoutBtn,
    "/editProfile" : editProfileBtn
}

export function activateBtn(page: string) {
    const Ping = document.getElementById("pongConnected") as HTMLButtonElement;
    if (Ping)
        Ping.addEventListener("click", (event: MouseEvent) => navigate(event, "/pong"));
    const Home = document.getElementById('home')  as HTMLButtonElement;
    if (Home)
        Home.addEventListener("click", (event: MouseEvent) => navigate(event, "/home"));
    const loginBtn = document.getElementById("loginButton")  as HTMLButtonElement;
    if (loginBtn)
        loginBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/login"));
    const logoutBtn = document.getElementById('logout')  as HTMLButtonElement;
    if (logoutBtn && connected)
        logoutBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/logout"));
    if (page in mapButton)
        mapButton[page]();
}

function connectBtn() {
    const button = document.getElementById("registerButton") as HTMLButtonElement;
    if (button)
        register(button);
}

function loginBtn() {
    const connect = document.getElementById('connectPageBtn') as HTMLButtonElement;
    if (connect)
        connect.addEventListener("click", (event: MouseEvent) => navigate(event, "/connect"));
    const button = document.getElementById("loginButton") as HTMLButtonElement;
    if (button)
        login(button);
}

function profileBtn() {
    const nickName = document.getElementById("profileNickName")!;
    const email = document.getElementById("profileEmail")!;
    // const container = document.getElementById('content') as HTMLElement;
    if (email && nickName)
        profile(/*container,*/ nickName, email);
    const editProfileBtn = document.getElementById("editProfileButton") as HTMLButtonElement;
    const addFriendBtn = document.getElementById("friendNameBtn") as HTMLButtonElement;
    const addFriendIpt = document.getElementById("friendNameIpt") as HTMLButtonElement;
    if (addFriendBtn && addFriendIpt) {
        addFriendBtn.addEventListener("click", () => addFriend(addFriendIpt.value));
    }

    if (editProfileBtn) {
        editProfileBtn.addEventListener("click", async () => console.log(await getFriendList()));
    }
}

function logoutBtn() {
    handleConnection(false);
    localStorage.removeItem('avatar');
    localStorage.removeItem('token');
    const avatar = document.getElementById("avatar") as HTMLImageElement;
    if (avatar)
        avatar.src = "../logout.png";
    const nickName = document.getElementById("nickName") as HTMLSpanElement;
    if (nickName)
        nickName.textContent = '';
}

function editProfileBtn() {

}