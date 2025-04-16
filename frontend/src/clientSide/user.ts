import {connected, handleConnection, loadPart, validateRegister} from "./front.js";
import {sseConnection} from "./serverSentEvent.js"

export function register(button: HTMLElement): void {
    button.addEventListener("click", async () => {
        const myForm = document.getElementById("myForm") as HTMLFormElement;
        const formData = new FormData(myForm);
        const data = Object.fromEntries(formData as unknown as Iterable<readonly any[]>);
        const response = await fetch('http://localhost:3000/user-management/sign-up', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        if (result.token) {
            localStorage.setItem('token', result.token);
            sseConnection(result.token);
            loadPart("/connected");
            handleConnection(true);
        } else {
            validateRegister(result.json);
        }
    });
}

export function login(button: HTMLElement): void {
    button.addEventListener("click", async () => {
        const myForm = document.getElementById("myForm") as HTMLFormElement;
        const formData = new FormData(myForm);
        const data = Object.fromEntries(formData as unknown as Iterable<readonly any[]>);
        const response = await fetch('http://localhost:3000/user-management/sign-in', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        if (response.ok) {
            localStorage.setItem('token', result.token);
            localStorage.setItem('nickName', result.nickName);
            sseConnection(result.token);
            loadPart("/connected");
            handleConnection(true);
        } else {
            const loginError = document.getElementById("LoginError") as HTMLSpanElement;
            loginError.textContent = result?.error ?? "An error occurred";
            loginError.classList.remove("hidden");
        }
    });
}

export async function profile(/*container: HTMLElement, */nickName: HTMLElement, email: HTMLElement) {
    try {
        const token = localStorage.getItem('token');
        if (!token) {
            console.error('token missing');
            return;
        }
        const response = await fetch('http://localhost:3000/user-management/getProfile', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'authorization': 'Bearer ' + token
            }
        })
        const data = await response.json();
        console.log(data);
        if (response.ok) {
            nickName.innerText = data.nickName;
            email.innerText = data.email;
        }
    }
    catch (err) {
        console.log(err)
    }

}

export async function addFriend(friendNickName: string) {
    try {
        const token = localStorage.getItem('token');
        const response = await fetch('http://localhost:3000/user-management/addFriend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'authorization': 'Bearer ' + token,
            },
            body: JSON.stringify({friendNickName: friendNickName}),
        });

        if (!response.ok) {
            const error = await response.json();
            console.log("USUS", error.message);
            return ;
        }
    } catch (error) {
        console.error(error);
    }
}

export function getAvatar() {
    if (!connected)
        return;
    const avatarImg = document.getElementById('avatar') as HTMLImageElement;
    const avatar = localStorage.getItem("avatar");
    if (!avatar) {
        console.log("No Avatar");
        avatarImg.src = '../login.png';
    } else {
        console.log("Avatar Found");
        avatarImg.src = avatar;
    }
}