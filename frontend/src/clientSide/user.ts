import {connected, handleConnection, loadPart, validateRegister} from "./front.js";
import {sseConnection} from "./serverSentEvent.js"

export function register(button: HTMLElement): void {
    button.addEventListener("click", async () => {
        const myForm = document.getElementById("myForm") as HTMLFormElement;
        const formData = new FormData(myForm);
        const data = Object.fromEntries(formData as unknown as Iterable<readonly any[]>);
        const response = await fetch('http://localhost:3000/user-management/register', {
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
        const response = await fetch('http://localhost:3000/user-management/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (response.ok) {
            const data = await response.json();
            localStorage.setItem('nickName', data.nickName);
            if (!data.token || data.token.empty) {
                console.log("2fa");
                return await loadPart("/2fa");
            }
            localStorage.setItem('token', data.token);
            await loadPart("/connected");
            handleConnection(true);
            await sseConnection(data.token);
        } else {
            const errorData = await response.json();
            const loginError = document.getElementById("LoginError") as HTMLSpanElement;
            loginError.textContent = errorData?.error ?? "An error occurred";
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
        if (response.ok) {
            nickName.innerText = data.nickName;
            email.innerText = data.email;
        }
    }
    catch (err) {
        console.log(err)
    }
}

export async function init2fa(): Promise<string | undefined> {
    try {
        const token = localStorage.getItem('token');
        if (!token) {
            console.error('token missing');
            return;
        }
        const response = await fetch('http://localhost:3000/user-management/2faInit', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'authorization': 'Bearer ' + token
            }
        })
        if (response.ok)
            return await response.json();
        const errorData =  await response.json();
        console.error(errorData);
        return undefined;
    } catch (err) {
        console.error(err)
    }
}

export async function verify2fa(secret: string): Promise<void> {
    try {
        const nickName = localStorage.getItem('nickname');
        const response = await fetch('http://localhost:3000/user-management/2faVerify', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({secret: secret, nickName: nickName})
        })
        if (response.ok)
            return await response.json();
        const errorData =  await response.json();
        console.error(errorData);
        return undefined;
    } catch (err) {
        console.error(err)
    }
}


export async function addFriend(friendNickName: string): Promise<boolean> {
    try {
        const token = localStorage.getItem('token');
        const response = await fetch('http://localhost:3000/user-management/addFriend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'authorization': 'Bearer ' + token,
            },
            body: JSON.stringify({friendNickName: friendNickName})
        });

        if (!response.ok) {
            const error = await response.json();
            console.error(error.error);
            return false;
        }
        return true;
    } catch (error) {
        console.error(error);
        return false;
    }
}

export async function removeFriend(friendNickName: string): Promise<boolean> {
    try {
        const token = localStorage.getItem('token');
        const response = await fetch('http://localhost:3000/user-management/removeFriend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'authorization': 'Bearer ' + token,
            },
            body: JSON.stringify({friendNickName: friendNickName}),
        });

        if (!response.ok) {
            const error = await response.json();
            console.error(error.error);
            return false;
        }
        return true;
    } catch (error) {
        console.error(error);
        return false
    }
}

export async function getFriendList(): Promise<string[] | undefined> {
    try {
        const token = localStorage.getItem('token');
        const response = await fetch('http://localhost:3000/user-management/friendList', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'authorization': 'Bearer ' + token,
            },
        });
        if (!response.ok) {
            const error = await response.json();
            console.error(error.error);
            return undefined;
        }
        return await response.json();
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

export async function createTournament() {
    try {
        const token = localStorage.getItem('token');
        const response = await fetch('http://localhost:3000/user-management/createTournament', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'authorization': 'Bearer ' + token,
            },
        });

        if (!response.ok) {
            const error = await response.json();
            console.error(error.error);
        }
    } catch (error) {
        console.error(error);
    }
}

export async function joinTournament() {
    try {
        const token = localStorage.getItem('token');
        const response = await fetch('http://localhost:3000/user-management/joinTournament', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'authorization': 'Bearer ' + token,
            },
        });

        if (!response.ok) {
            const error = await response.json();
            console.error(error.error);
        }
    } catch (error) {
        console.error(error);
    }
}

export async function getTournamentList(): Promise<string[] | undefined> {
    try {
        const token = localStorage.getItem('token');
        const response = await fetch('http://localhost:3000/user-management/getTournamentList', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'authorization': 'Bearer ' + token,
            },
        });
        if (!response.ok) {
            const error = await response.json();
            console.error(error.error);
            return undefined;
        }
        return await response.json();
    } catch (error) {
        console.error(error);
    }
}

export async function launchTournament() {
    try {
        const token = localStorage.getItem('token');
        const response = await fetch('http://localhost:3000/user-management/launchTournament', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'authorization': 'Bearer ' + token,
            },
        });

        if (!response.ok) {
            const error = await response.json();
            console.error(error.error);
        }
    } catch (error) {
        console.error(error);
    }
}

