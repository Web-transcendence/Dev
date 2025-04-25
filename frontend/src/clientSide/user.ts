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

export interface Person {
    name: string;
    // email: string;
    imageUrl: string;
}

type FriendList = {
    acceptedNickName: string[];
    pendingNickName: string[];
    receivedNickName: string[];
};

export async function friendList(): Promise<void> {
    try {
        const friendlist = await getFriendList() as FriendList;
        console.log("RECEIVED", friendlist.receivedNickName);
        console.log("PENDING", friendlist.pendingNickName);
        console.log("ACCEPTED", friendlist.acceptedNickName);
        const noFriend = document.getElementById("noFriend") as HTMLHeadingElement;
        if (!friendlist || !friendlist.receivedNickName.length) {
            if (noFriend)
                noFriend.classList.remove("hidden");
            console.log("friendlist is undefined");
        } else {
            const receivedPeople: Person[] = friendlist.receivedNickName.map(nickname => ({
                name: nickname,
                imageUrl: '../images/login.png'
            }));
            const receivedList = document.getElementById("receivedList");
            const receivedTemplate = document.getElementById("receivedTemplate") as HTMLTemplateElement;

            if (receivedList && receivedTemplate) {
                receivedList.innerHTML = '';
                receivedPeople.forEach(person => {
                    const clone = receivedTemplate.content.cloneNode(true) as HTMLElement;
                    const img = clone.querySelector("img")!;
                    const name = clone.querySelector(".name")!;

                    img.src = person.imageUrl;
                    img.alt = person.name;
                    name.textContent = person.name;

                    receivedList.appendChild(clone);
                });
            }
        }
        if (!friendlist || !friendlist.pendingNickName.length) {
            if (noFriend)
                noFriend.classList.remove("hidden");
            console.log("friendlist is undefined");
        } else {
            const requestPeople: Person[] = friendlist.pendingNickName.map(nickname => ({
                name: nickname,
                imageUrl: '../images/login.png'
            }));
            const requestList = document.getElementById("requestList");
            const requestTemplate = document.getElementById("requestTemplate") as HTMLTemplateElement;

            if (requestList && requestTemplate) {
                requestList.innerHTML = '';
                requestPeople.forEach(person => {
                    const clone = requestTemplate.content.cloneNode(true) as HTMLElement;
                    const img = clone.querySelector("img")!;
                    const name = clone.querySelector(".name")!;

                    img.src = person.imageUrl;
                    img.alt = person.name;
                    name.textContent = person.name;

                    requestList.appendChild(clone);
                });
            }
        }
        if (!friendlist || !friendlist.acceptedNickName.length) {
            if (noFriend)
                noFriend.classList.remove("hidden");
            console.log("friendlist is undefined");
        } else {
            const acceptedPeople: Person[] = friendlist.acceptedNickName.map(nickname => ({
                name: nickname,
                imageUrl: '../images/login.png'
            }));
            const acceptedList = document.getElementById("acceptedList");
            const acceptedTemplate = document.getElementById("acceptedTemplate") as HTMLTemplateElement;

            if (acceptedList && acceptedTemplate) {
                acceptedList.innerHTML = '';
                acceptedPeople.forEach(person => {
                    const clone = acceptedTemplate.content.cloneNode(true) as HTMLElement;
                    const img = clone.querySelector("img")!;
                    const name = clone.querySelector(".name")!;

                    img.src = person.imageUrl;
                    img.alt = person.name;
                    name.textContent = person.name;

                    acceptedList.appendChild(clone);
                });
            }
        }
    } catch (err) {
        console.error("Erreur dans friendList():", err);
    }
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

export async function init2fa(): Promise<string | any> {
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
            return await response.text();
        const errorData =  await response.json();
        return errorData;
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
        //Clear input after use
        const input = document.getElementById('friendNameIpt') as HTMLInputElement;
        input.value = '';
        input.focus();
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

export async function getFriendList(): Promise<FriendList | undefined> {
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

const toBase64 = (file: any) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
});

export async function setAvatar(target: HTMLInputElement) {
    try {
        if (target.files && target.files[0]) {
            const file: File = target.files[0];
            const base64File = await toBase64(file);

            const token = localStorage.getItem('token');
            const response = await fetch('http://localhost:3000/user-management/updatePicture', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'authorization': 'Bearer ' + token,
                },
                body: JSON.stringify({pictureURL: base64File})
            });

            if (!response.ok) {
                const error = await response.json();
                console.error(error.error);
            }
        }
    } catch (err) {
        console.log(err);
    }
}

export async function getAvatar() {
    const avatarImg = document.getElementById('avatar') as HTMLImageElement;
    try {
        const token = localStorage.getItem('token');
        const response = await fetch('http://localhost:3000/user-management/getPicture', {
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
        const img = await response.json();
        if (img.url)
            avatarImg.src = img.url;
        else
            avatarImg.src = '../images/login.png';

    } catch (error) {
        console.error(error);
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
