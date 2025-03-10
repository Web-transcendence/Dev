document.addEventListener("DOMContentLoaded", () => {
    const aboutBtn = document.getElementById("about")!;
    const contactBtn = document.getElementById("contact")!;
    const registerBtn = document.getElementById("register")!;
    aboutBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/about"));
    contactBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/contact"));
    registerBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/register"));
    loadPart(window.location.pathname);
});

window.onpopstate = () => {
    loadPart(window.location.pathname);
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
    button.addEventListener("click", async (event) => {
        const myForm = document.getElementById("myForm") as HTMLFormElement;
        const formData = new FormData(myForm);
        const data: Record<string, unknown> = Object.fromEntries(formData as unknown as Iterable<readonly any[]>);
        const response = await fetch('http://localhost:8000/user-management/sign-up', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        if (result.redirect) {
            const res = await fetch(`${result.redirect}`, {});
            const newElement = document.createElement('div');
            newElement.className = 'tag';
            if (!res.ok)
                throw new Error("Page not found: redirect missing.");
            const html = await res.text();
            container.innerHTML = '';
            newElement.innerHTML = html;
            container.appendChild(newElement);
        } else
            validateForm(result);
    });
}

function validateForm(result: { name: string; email: string; password: string}): void {
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

