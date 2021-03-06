use rand::Rng;
use std::rc::Rc;

const F32_INFINITY: f32 = f32::INFINITY;
const PI: f32 = 3.1415926535897932385f32;

fn degrees_to_radians(degrees: f32) -> f32 {
	return degrees * PI / 180.0f32;
}

fn random_float() -> f32 {
	let mut rng = rand::thread_rng();
	rng.gen::<f32>()
}

fn random_float_min_max(min: f32, max: f32) -> f32 {
	min + (max - min) * random_float()
}

fn clamp(x: f32, min: f32, max: f32) -> f32 {
	if x < min { return min; }
	if max < x { return max; }
	return x;
}
fn clampv(v: &Vec3, min: f32, max: f32) -> Vec3 {
	Vec3::new(clamp(v.x(), min, max), clamp(v.y(), min, max), clamp(v.z(), min, max))
}
fn clampv_gamma(v: &Vec3, min: f32, max: f32) -> Vec3 {
	Vec3::new(clamp(v.x().sqrt(), min, max), clamp(v.y().sqrt(), min, max), clamp(v.z().sqrt(), min, max))
}

fn write_color(clr: &Vec3, samples_count: i32) {
	let scale: f32 = 1.0f32 / (samples_count as f32);
	let rgb: Vec3 = *clr * scale;
	println!("{}", clampv_gamma(&rgb, 0.0f32, 0.9999f32));
}
#[derive(Copy, Clone)]
struct Vec3 {
	e: [f32; 3]
}

struct Ray {
	m_origin: Vec3,
	m_direction: Vec3,
}

struct HitRecord {
	m_point: Vec3,
	m_normal: Vec3,
	m_material: Rc<dyn Material>,
	m_t: f32,
	m_frontface: bool,
}

trait Hittable {
	fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> (bool, HitRecord);
}

trait Material {
	fn scatter(&self, r_in: &Ray, rec: &HitRecord) -> (bool, Vec3, Ray);
}

struct Sphere {
	m_center: Vec3,
	m_radius: f32,
	m_material: Rc<dyn Material>,
}

type IHittable = Box<dyn Hittable>;
type HittableVec = Vec<IHittable>;

struct HittableList {
	m_objects: HittableVec,
}

struct Camera {
	m_origin: Vec3,
	m_lower_left_corner: Vec3,
	m_horizontal: Vec3,
	m_vertical: Vec3,
}

#[derive(Copy, Clone)]
struct Lambertian {
	m_albedo: Vec3
}
struct Metal {
	m_albedo: Vec3,
	m_fuzz: f32,
}
struct Dielectric {
	m_ior: f32
}

impl Vec3 {
	fn new(x: f32, y: f32, z:f32) -> Vec3 {
		Vec3{ e:[x, y, z] }
	}
	fn x(&self) -> f32 {
		self.e[0]
	}
	fn y(&self) -> f32 {
		self.e[1]
	}
	fn z(&self) -> f32 {
		self.e[2]
	}
	fn r(&self) -> u8 {
		(self.e[0] * 255.999f32) as u8
	}
	fn g(&self) -> u8 {
		(self.e[1] * 255.999f32) as u8
	}
	fn b(&self) -> u8 {
		(self.e[2] * 255.999f32) as u8
	}
	fn length_squared(&self) -> f32 {
		dot(&self, &self)
	}
	fn length(&self) -> f32 {
		self.length_squared().sqrt()
	}
	fn random() -> Vec3 {
		Vec3 { e: [random_float(), random_float(), random_float()] }
	}
	fn random_range(min: f32, max: f32) -> Vec3 {
		Vec3 { e: [random_float_min_max(min, max), random_float_min_max(min, max), random_float_min_max(min, max)] }
	}
	fn random_in_unit_sphere() -> Vec3 {
		loop {
			let p: Vec3 = Vec3::random_range(-1.0f32, 1.0f32);
			if 1.0f32 <= p.length_squared() { continue; }
			return p;
		}
	}
	fn random_unit_vector() -> Vec3 {
		unit_vector(Vec3::random_in_unit_sphere())
	}
	fn near_zero(&self) -> bool {
		const s: f32 = 1e-8f32;
		self.e[0].abs() < s && self.e[1].abs() < s && self.e[2].abs() < s
	}
}

fn dot(u: &Vec3, v: &Vec3) -> f32 {
	u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2]
}
fn cross(u: &Vec3, v: &Vec3) -> Vec3 {
	Vec3{ e: [
		u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0],
	] }
}
fn reflect(v: &Vec3, n: &Vec3) -> Vec3 {
	*v - *n * (2.0f32 * dot(v, n))
}
fn refract(v: &Vec3, n: &Vec3, etai_over_etat: f32) -> Vec3 {
	let uv: Vec3 = *v;
	let cos_theta: f32 = dot(&-uv, n).min(1.0f32);
	let r_out_perp: Vec3 = (uv + *n * cos_theta) * etai_over_etat;
	let r_out_parallel: Vec3 = *n * -(1.0f32 - r_out_perp.length_squared()).abs().sqrt();
	r_out_perp + r_out_parallel
}
fn unit_vector(v: Vec3) -> Vec3 {
	v / v.length()
}

use std::fmt;
impl fmt::Display for Vec3 {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "{0} {1} {2}", self.r(), self.g(), self.b())
	}
}
impl fmt::Debug for Vec3 {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "{0} {1} {2}", self.x(), self.y(), self.z())
	}
}

use std::ops::{ Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign, Neg };
impl Neg for Vec3 {
	type Output = Self;
	fn neg(self) -> Self {
		Self{ e: [-self.e[0], -self.e[1], -self.e[2]] }
	}
}
impl Add<Vec3> for Vec3 {
	type Output = Vec3;
	fn add(self, other: Vec3) -> Self {
		Self{ e: [self.e[0] + other.e[0], self.e[1] + other.e[1], self.e[2] + other.e[2]] }
	}
}
impl Add<f32> for Vec3 {
	type Output = Vec3;
	fn add(self, other: f32) -> Self {
		Self{ e: [self.e[0] + other, self.e[1] + other, self.e[2] + other] }
	}
}
impl AddAssign<Vec3> for Vec3 {
	fn add_assign(&mut self, other: Self) {
		*self = Self{ e: [self.e[0] + other.e[0], self.e[1] + other.e[1], self.e[2] + other.e[2]] };
	}
}
impl AddAssign<f32> for Vec3 {
	fn add_assign(&mut self, other: f32) {
		*self = Self{ e: [self.e[0] + other, self.e[1] + other, self.e[2] + other] };
	}
}
impl Sub<Vec3> for Vec3 {
	type Output = Self;
	fn sub(self, other: Vec3) -> Self {
		Self{ e: [self.e[0] - other.e[0], self.e[1] - other.e[1], self.e[2] - other.e[2]] }
	}
}
impl Sub<f32> for Vec3 {
	type Output = Self;
	fn sub(self, other: f32) -> Self {
		Self{ e: [self.e[0] - other, self.e[1] - other, self.e[2] - other] }
	}
}
impl SubAssign<Vec3> for Vec3 {
	fn sub_assign(&mut self, other: Vec3) {
		*self = Self{ e: [self.e[0] - other.e[0], self.e[1] - other.e[1], self.e[2] - other.e[2]] };
	}
}
impl SubAssign<f32> for Vec3 {
	fn sub_assign(&mut self, other: f32) {
		*self = Self{ e: [self.e[0] - other, self.e[1] - other, self.e[2] - other] };
	}
}
impl Mul<Vec3> for Vec3 {
	type Output = Self;
	fn mul(self, other: Vec3) -> Self {
		Self{ e: [self.e[0] * other.e[0], self.e[1] * other.e[1], self.e[2] * other.e[2]] }
	}
}
impl Mul<f32> for Vec3 {
	type Output = Self;
	fn mul(self, other: f32) -> Self {
		Self{ e: [self.e[0] * other, self.e[1] * other, self.e[2] * other] }
	}
}
impl MulAssign<Vec3> for Vec3 {
	fn mul_assign(&mut self, other: Vec3) {
		*self = Self{ e: [self.e[0] * other.e[0], self.e[1] * other.e[1], self.e[2] * other.e[2]] };
	}
}
impl MulAssign<f32> for Vec3 {
	fn mul_assign(&mut self, other: f32) {
		*self = Self{ e: [self.e[0] * other, self.e[1] * other, self.e[2] * other] };
	}
}
impl Div<Vec3> for Vec3 {
	type Output = Self;
	fn div(self, other: Vec3) -> Self {
		Self{ e: [self.e[0] / other.e[0], self.e[1] / other.e[1], self.e[2] / other.e[2]] }
	}
}
impl Div<f32> for Vec3 {
	type Output = Self;
	fn div(self, other: f32) -> Self {
		let inv : f32 = 1.0f32 / other;
		Self{ e: [self.e[0] * inv, self.e[1] * inv, self.e[2] * inv] }
	}
}
impl DivAssign<Vec3> for Vec3 {
	fn div_assign(&mut self, other: Vec3) {
		*self = Self{ e: [self.e[0] / other.e[0], self.e[1] / other.e[1], self.e[2] / other.e[2]] };
	}
}
impl DivAssign<f32> for Vec3 {
	fn div_assign(&mut self, other: f32) {
		let inv : f32 = 1.0f32 / other;
		*self = Self{ e: [self.e[0] * inv, self.e[1] * inv, self.e[2] * inv] };
	}
}

impl Ray {
	fn new(origin: &Vec3, direction: &Vec3) -> Ray {
		Ray{ m_origin: *origin, m_direction: *direction }
	}
	fn origin(&self) -> Vec3 {
		self.m_origin
	}
	fn direction(&self) -> Vec3 {
		self.m_direction
	}
	fn at(&self, t: f32) -> Vec3 {
		self.m_origin + self.m_direction * t
	}
}

impl Sphere {
	fn new(center: &Vec3, radius: f32, material: &Rc<dyn Material>) -> Sphere {
		Sphere{ m_center: *center, m_radius: radius, m_material: material.clone() }
	}
}

impl HitRecord {
	fn new(root: f32, point: Vec3, material: &Rc<dyn Material>) -> HitRecord {
		HitRecord {
			m_point: point,
			m_normal: Vec3::new(0.0f32, 0.0f32, 0.0f32),
			m_t: root,
			m_material: material.clone(),
			m_frontface: false,
		}
	}
	fn default() -> HitRecord {
		static black_clr: Vec3 = Vec3{e:[0.0f32, 0.0f32, 0.0f32]};
		static black_lambertian: Lambertian = Lambertian { m_albedo: black_clr };
		HitRecord {
			m_point: Vec3::new(0.0f32, 0.0f32, 0.0f32),
			m_normal: Vec3::new(0.0f32, 0.0f32, 0.0f32),
			m_t: 0.0f32,
			m_material: Rc::new(black_lambertian),
			m_frontface: false,
		}
	}
	fn set_face_normal(&mut self, r: &Ray, n: &Vec3) {
		let d: Vec3 = r.direction();
		self.m_frontface = dot(&d, n) < 0.0f32;
		self.m_normal = if self.m_frontface { *n } else { -(*n) };
	}
}

impl Hittable for Sphere {
	fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> (bool, HitRecord) {
		let oc: Vec3 = r.origin() - self.m_center;
		let a: f32 = r.direction().length_squared();
		let half_b: f32 = dot(&oc, &r.direction());
		let c: f32 = oc.length_squared() - self.m_radius * self.m_radius;
		
		let discr: f32 = half_b * half_b - a * c;
		if discr < 0.0f32 {
			return (false, HitRecord::default());
		}
		
		let sqrtd: f32 = discr.sqrt();
		let mut root: f32 = -(half_b + sqrtd) / a;
		if root < t_min || t_max < root {
			root = (-half_b + sqrtd) / a;
			if root < t_min || t_max < root {
				return (false, HitRecord::default());
			}
		}
		
		let point: Vec3 = r.at(root);
		let outward_normal: Vec3 = (point - self.m_center) / self.m_radius;
		let mut rec = HitRecord::new(root, point, &self.m_material);
		rec.set_face_normal(&r, &outward_normal);
		return (true, rec);
	}
}

impl HittableList {
	fn new() -> HittableList {
		HittableList {
			m_objects: Vec::new()
		}
	}
}

impl Hittable for HittableList {
	fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> (bool, HitRecord) {
		let mut temp_rec: HitRecord = HitRecord::default();
		let mut hit_anything: bool = false;
		let mut closest_so_far: f32 = t_max;
		
		for object in &self.m_objects {
			let results = object.hit(r, t_min, closest_so_far);
			if results.0 {
				hit_anything = true;
				closest_so_far = temp_rec.m_t;
				temp_rec = results.1;
			}
		}
		
		(hit_anything, temp_rec)
	}
}

impl Camera {
	fn new() -> Camera {
		let aspect_ratio: f32 = 16.0f32 / 9.0f32;
		let viewport_height: f32 = 2.0f32;
		let viewport_width: f32 = viewport_height * aspect_ratio;
		let focal_length: f32 = 1.0f32;
		let origin: Vec3 = Vec3::new(0.0f32, 0.0f32, 0.0f32);
		let horizontal: Vec3 = Vec3::new(viewport_width, 0.0f32, 0.0f32);
		let vertical: Vec3 = Vec3::new(0.0f32, viewport_height, 0.0f32);
		
		Camera {
			m_origin: origin,
			m_horizontal: horizontal,
			m_vertical: vertical,
			m_lower_left_corner: origin - horizontal * 0.5f32 - vertical * 0.5f32 - Vec3::new(0.0f32, 0.0f32, focal_length),
		}
	}
	fn get_ray(&self, u: f32, v: f32) -> Ray {
		Ray::new(&self.m_origin, &(self.m_lower_left_corner + self.m_horizontal * u + self.m_vertical * v - self.m_origin))
	}
}

impl Lambertian {
	fn new(albedo: &Vec3) -> Lambertian {
		Lambertian { m_albedo: *albedo }
	}
}

impl Metal {
	fn new(albedo: &Vec3, fuzz: f32) -> Metal {
		Metal { m_albedo: *albedo, m_fuzz: fuzz }
	}
}

impl Dielectric {
	fn new(index_of_refraction: f32) -> Dielectric {
		Dielectric{ m_ior: index_of_refraction }
	}
	fn reflectance(cosine: f32, ref_idx: f32) -> f32 {
		let mut r0: f32 = (1.0f32 - ref_idx) / (1.0f32 + ref_idx);
		r0 = r0 * r0;
		r0 + (1.0f32 - r0) * (1.0f32 - cosine).powf(5.0f32)
	}
}

impl Material for Lambertian {
	fn scatter(&self, _r_in: &Ray, rec: &HitRecord) -> (bool, Vec3, Ray) {
		let mut scatter_direction: Vec3 = rec.m_normal + Vec3::random_unit_vector();
		if scatter_direction.near_zero() {
			scatter_direction = rec.m_normal;
		}
		return (true, self.m_albedo, Ray::new(&rec.m_point, &scatter_direction));
	}
}

impl Material for Metal {
	fn scatter(&self, r_in: &Ray, rec: &HitRecord) -> (bool, Vec3, Ray) {
		let reflected: Vec3 = reflect(&unit_vector(r_in.direction()), &rec.m_normal);
		let scattered: Vec3 = reflected + Vec3::random_unit_vector() * self.m_fuzz;
		return (0.0f32 < dot(&reflected, &rec.m_normal), self.m_albedo, Ray::new(&rec.m_point, &scattered));
	}
}

impl Material for Dielectric {
	fn scatter(&self, r_in: &Ray, rec: &HitRecord) -> (bool, Vec3, Ray) {
		let refraction_ratio: f32 = if rec.m_frontface { 1.0f32 / self.m_ior } else { self.m_ior };
		let unit_direction: Vec3 = unit_vector(r_in.direction());
		let cos_theta: f32 = dot(&-unit_direction, &rec.m_normal).min(1.0f32);
		let sin_theta: f32 = (1.0f32 - cos_theta * cos_theta).sqrt();
		let cannot_refract: bool = 1.0f32 < refraction_ratio * sin_theta;
		let mut direction: Vec3 = Vec3::new(0.0f32, 0.0f32, -1.0f32);
		if cannot_refract || random_float() < Dielectric::reflectance(cos_theta, refraction_ratio) {
			direction = reflect(&unit_direction, &rec.m_normal);
		} else {
			direction = refract(&unit_direction, &rec.m_normal, refraction_ratio);
		}
		(true, Vec3::new(1.0f32, 1.0f32, 1.0f32), Ray::new(&rec.m_point, &direction))
	}
}

fn ray_color(r: &Ray, world: &dyn Hittable, depth: i32) -> Vec3 {
	if depth <= 0 { return Vec3::new(0.0f32, 0.0f32, 0.0f32); }
	let hit_result = world.hit(r, 0.001f32, F32_INFINITY);
	if hit_result.0 {
		let rec: HitRecord = hit_result.1;
		let scatter_res = rec.m_material.scatter(&r, &rec);
		if scatter_res.0 {
			return scatter_res.1 * ray_color(&scatter_res.2, world, depth - 1);
		}
		return Vec3::new(0.0f32, 0.0f32, 0.0f32);
	}
	let unit_direction: Vec3 = unit_vector(r.direction());
	let t: f32 = 0.5f32 * (unit_direction.y() + 1.0f32);
	return Vec3::new(1.0f32, 1.0f32, 1.0f32) * (1.0f32 - t) + Vec3::new(0.5f32, 0.7f32, 1.0f32) * t;
}

fn main() {	
	//image
	let aspect_ratio: f32 = 16.0f32 / 9.0f32;
	let image_width: u32 = 400;
	let image_height: u32 = ((image_width as f32) / aspect_ratio) as u32;
	let image_width_inv : f32 = 1.0f32 / (image_width - 1u32) as f32;
	let image_height_inv : f32 = 1.0f32 / (image_height - 1u32) as f32;
	let samples_count: i32 = 100;
	let max_depth: i32 = 50;
	
	//world
	let material_ground = Box::new(Lambertian::new(&Vec3::new(0.8f32, 0.8f32, 0.0f32)));
	let material_center = Box::new(Lambertian::new(&Vec3::new(0.1f32, 0.2f32, 0.5f32)));
	let material_left = Box::new(Dielectric::new(1.5f32));
	let material_right = Box::new(Metal::new(&Vec3::new(0.8f32, 0.6f32, 0.2f32), 0.0f32));
	
	let material_ground_rc = Rc::from(material_ground as Box<dyn Material>);
	let material_center_rc = Rc::from(material_center as Box<dyn Material>);
	let material_left_rc = Rc::from(material_left as Box<dyn Material>);
	let material_right_rc = Rc::from(material_right as Box<dyn Material>);
	
	let sphere0 = Box::new(Sphere::new(&Vec3::new( 0.0f32, -100.5f32, -1.0f32), 100.0f32, &material_ground_rc));
	let sphere1 = Box::new(Sphere::new(&Vec3::new( 0.0f32,    0.0f32, -1.0f32),   0.5f32, &material_center_rc));
	let sphere2 = Box::new(Sphere::new(&Vec3::new(-1.0f32,    0.0f32, -1.0f32),   0.5f32, &material_left_rc));
	let sphere3 = Box::new(Sphere::new(&Vec3::new( 1.0f32,    0.0f32, -1.0f32),   0.5f32, &material_right_rc));
	let mut world: HittableList = HittableList::new();
	world.m_objects.push(sphere1);
	world.m_objects.push(sphere2);
	world.m_objects.push(sphere3);
	world.m_objects.push(sphere0);
	
	//camera
	let cam: Camera = Camera::new();
	
	//render
	println!("P3");
	println!("{0} {1}", image_width, image_height);
	println!("255");
	
	for j in (0u32..image_height).rev() {
		eprint!("\rScanlines remaining: {0}", j);
		for i in 0..image_width {
			let mut pixel_color: Vec3 = Vec3::new(0.0f32, 0.0f32, 0.0f32);
			for _s in 0..samples_count {
				let u = (i as f32 + random_float()) * image_width_inv;
				let v = (j as f32 + random_float()) * image_height_inv;
				let r: Ray = cam.get_ray(u, v);
				pixel_color += ray_color(&r, &world, max_depth);
			}
			write_color(&pixel_color, samples_count);
		}
	}
	eprintln!("Done!");
}
