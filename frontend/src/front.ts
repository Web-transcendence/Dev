document.addEventListener("DOMContentLoaded", () => {
    const aboutBtn = document.getElementById("about")!;
    const contactBtn = document.getElementById("contact")!;
    const registerBtn = document.getElementById("register")!;
    const loginBtn = document.getElementById("login")!;
    const profileBtn = document.getElementById("profile")!;
    const disconnectBtn = document.getElementById("disconnect")!;
    isConnected();

    profileBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/profile"))
    aboutBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/about"));
    contactBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/contact"));
    registerBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/register"));
    loginBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/login"));
    disconnectBtn.addEventListener("click", (event: MouseEvent) => {
        localStorage.removeItem("token");
        isConnected();
        navigate(event, "/");

    });
    // loadPart(window.location.pathname);
});

window.onpopstate = () => {
    // loadPart(window.location.pathname);
};

function navigate(event: MouseEvent, path: string): void {
    event.preventDefault();
    window.history.pushState({}, "", path);
    loadPart(path);
}

async function loadPart(page: string): Promise<void> {
    const container = document.getElementById('content') as HTMLElement;
    if (!page || page === "/") {
        container.innerHTML = '';
        container.innerHTML = "<p>Choisissez une option pour charger le contenu.</p>";
        return;
    }
    try {
        await insert_tag(`part${page}`);
        if (page === "/register") {
            const button = document.getElementById("registerButton")!;
            if (button)
                register(container, button);
        }
        if (page === "/login") {
            const button = document.getElementById("loginButton")!;
            if (button)
                login(container, button);
        }
        if (page === "/profile") {

            const name = document.getElementById("profileName")!;
            const email = document.getElementById("profileEmail")!;
            if (email && name)
                profile(container, name, email);
        }
    } catch (error) {
        console.error(error);
        container.innerHTML = '';
        container.innerHTML = "<h2>404 - Page non trouvée</h2>";
    }
}

async function insert_tag(url: string): Promise<void>{
    const container = document.getElementById('content') as HTMLElement;
    const res = await fetch(url);
    const newElement = document.createElement('div');
    newElement.className = 'tag';
    if (!res.ok)
        throw Error("Page not found: element missing.");
    const html = await res.text();
    if (html.includes(container.innerHTML))
        return;
    container.innerHTML = '';
    newElement.innerHTML = html;
    container.appendChild(newElement);
}

function register(container: HTMLElement, button: HTMLElement): void {
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
            isConnected();
            const res = await fetch('/part/connected');
            const newElement = document.createElement('div');
            newElement.className = 'tag';
            if (!res.ok)
                throw Error("Page not found: element missing.");
            const html = await res.text();
            if (html.includes(container.innerHTML))
                return;
            container.innerHTML = '';
            newElement.innerHTML = html;
            container.appendChild(newElement);
        } else {
            console.log("result.token");
            const errors = result.json;
            validateRegister(errors.json);
        }
    });
}

function login(container: HTMLElement, button: HTMLElement): void {
    button.addEventListener("click", async () => {
        const myForm = document.getElementById("myForm") as HTMLFormElement;
        const formData = new FormData(myForm);
        const data = Object.fromEntries(formData as unknown as Iterable<readonly any[]>);
        console.log(data);
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
            isConnected()
            localStorage.setItem('name', result.name);
            // localStorage.setItem('avatar', result.avatar);
            const res = await fetch('/part/connected');
            const newElement = document.createElement('div');
            newElement.className = 'tag';
            if (!res.ok)
                throw Error("Page not found: element missing.");
            const html = await res.text();
            if (html.includes(container.innerHTML))
                return;
            container.innerHTML = '';
            newElement.innerHTML = html;
            container.appendChild(newElement);
        } else {
            const loginError = document.getElementById("LoginError") as HTMLSpanElement;
            // if (!loginError.classList.contains("hidden"))
            loginError.textContent = result?.error ?? "An error occurred";
            loginError.classList.remove("hidden");
        }
    });
}


async function profile(container: HTMLElement, name: HTMLElement, email: HTMLElement) {
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
            name.innerText = data.name;
            email.innerText = data.email;
        }

    }
    catch (err) {

    }
}

function validateRegister(result: { name: string; email: string; password: string}): void {
    const nameError = document.getElementById("nameError") as HTMLSpanElement;
    const emailError = document.getElementById("emailError") as HTMLSpanElement;
    const passwordError = document.getElementById("passwordError") as HTMLSpanElement;
    if (result.name)
        nameError.classList.remove("hidden");
    else {
        if (!nameError.classList.contains("hidden")) {
            nameError.classList.add("hidden");
        }
    }
    if (result.email)
        emailError.classList.remove("hidden");
    else {
        if (!emailError.classList.contains("hidden")) {
            emailError.classList.add("hidden");
        }
    }
    if (result.password)
        passwordError.classList.remove("hidden");
    else {
        if (!passwordError.classList.contains("hidden")) {
            passwordError.classList.add("hidden");
        }
    }
}



function isConnected() {
    if (localStorage.getItem("token")) {
        document.getElementById("profile")!.classList.remove("hidden");
        document.getElementById("register")!.classList.add("hidden");
        document.getElementById("login")!.classList.add("hidden");
        document.getElementById("disconnect")!.classList.remove("hidden");
    }
    else {
        document.getElementById("profile")!.classList.add("hidden");
        document.getElementById("register")!.classList.remove("hidden");
        document.getElementById("login")!.classList.remove("hidden");
        document.getElementById("disconnect")!.classList.add("hidden");
    }
}

